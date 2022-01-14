#include "slife/slife_handler.h"
#include <functional>

using namespace std;
using namespace torch;
using namespace lietorch;

SlifeHandler::SlifeHandler(const SlifeHandler::TensorPublisherExtra &_tensorPublisher):
	tensorPublishExtraCallback(_tensorPublisher),
	tensorPublishCallback(bind (_tensorPublisher,
						   placeholders::_1,
						   placeholders::_2,
						   std::vector<uint8_t> ()))
{
	flags.addFlag ("initialized", true);

	seq.pcl = -1;
	seq.groundTruth = -1;
}


template<>
Position SlifeHandler::poseTensorToGroup<Position> (const Tensor &grounTruthTensor) const {
	return Position (grounTruthTensor.slice (0,0,LIETORCH_POSITION_DIM));
}

template<>
Quaternion SlifeHandler::poseTensorToGroup<Quaternion> (const Tensor &grounTruthTensor) const {
	return Quaternion (grounTruthTensor.slice (0,LIETORCH_POSITION_DIM));
}

template<>
Pose SlifeHandler::poseTensorToGroup<Pose> (const Tensor &groundTruthTensor) const {
	return Pose (groundTruthTensor);
}

void SlifeHandler::updateGroundTruth (const Tensor &groundTruthTensor) {
	seq.groundTruth++;
	groundTruthTracker->updateGroundTruth (
				poseTensorToGroup<TargetGroup> (groundTruthTensor));
}

void SlifeHandler::performOptimization (const Tensor &pointcloud)
{
	seq.pcl++;
	optimizer->costFunction()->updatePointcloud (pointcloud);

	if (optimizer->isReady())
	{
		TargetGroup estimate = optimizer->optimize();
		TargetGroup relativeGroundTruth = groundTruthTracker->getRelativeGroundTruth ();
		cout << "optimizing with seq.pcl = " << seq.pcl << "; seq.groundTruth = " << seq.groundTruth << endl;
		COUTN(relativeGroundTruth);
		auto history = optimizer->getHistory ();

		tensorPublishCallback (OUTPUT_ESTIMATE, estimate.coeffs);
		tensorPublishExtraCallback (OUTPUT_DEBUG_1, historyToTensor (history), {params.targetOptimizationGroup});
		tensorPublishExtraCallback (OUTPUT_DEBUG_2, computeHistoryError (history, relativeGroundTruth), {params.targetOptimizationGroup});
	}
}

Tensor SlifeHandler::computeHistoryError (const std::vector<TargetGroup> &historyVector, const TargetGroup &groundTruth)
{
	Tensor errorHistory = torch::empty ({(long int) historyVector.size (),(long int) TargetGroup::Tangent::Dim}, kFloat);
	int i = 0;
	
	for (const TargetGroup &curr : historyVector) {
		errorHistory[i] = (curr - groundTruth).coeffs;
		i++;
	}

	return errorHistory;
}

Tensor SlifeHandler::historyToTensor (const std::vector<TargetGroup> &historyVector)
{
	Tensor historyTensor = torch::empty({(long int) historyVector.size(), TargetGroup::Dim}, kFloat);
	int i = 0;

	for (auto curr : historyVector) {
		historyTensor[i] = curr.coeffs;
		i++;
	}

	return historyTensor;
}

void SlifeHandler::test ()
{
	Test::Type testWhat = tester->getType();
	Tensor testValues;

	testValues = optimizer->costFunction()->test (testWhat);

	if (testValues.numel ())
		tester->publishRangeTensor (testWhat, testValues);
}


int SlifeHandler::synchronousActions ()
{
	if (optimizer->costFunction()->isReady())
		test ();

	return 0;
}

void SlifeHandler::init (XmlRpc::XmlRpcValue &xmlParams)
{
	typename PointcloudMatchOptimizer<TargetGroup>::Params::Ptr optimizerParams = getOptimizerParams (xmlParams["optimizer"]);
	typename PointcloudMatch<TargetGroup>::Params::Ptr costFunctionParams = getCostFunctionParams (xmlParams["optimizer"]["cost"]);
	Landscape::Params::Ptr landscapeParams = getLandscapeParams(xmlParams["landscape"]);

	params = getHandlerParams (xmlParams);

	optimizer = make_shared<PointcloudMatchOptimizer<TargetGroup>> (optimizerParams,
													    make_shared<PointcloudMatch<TargetGroup>> (landscapeParams,
																						  costFunctionParams));
	groundTruthTracker = make_shared<GroundTruthTracker> (params.groundTruthTracker);

	flags.set ("initialized");
}

bool SlifeHandler::isSyntheticPcl () const {
	return params.syntheticPcl;
}

bool SlifeHandler::isReady() const {
	return flags.isReady ();
}

SlifeHandler::Params SlifeHandler::getHandlerParams (XmlRpc::XmlRpcValue &xmlParams)
{
	Params params;

	params.syntheticPcl = paramBool (xmlParams, "synthetic_pcl");
	params.targetOptimizationGroup = paramEnum<TargetOptimizationGroup> (xmlParams, "target_optimization_group",{"position","quaternion_r4","quaternion","pose_r4","pose","dual_quaternion"});
	params.groundTruthTracker.queueLength = paramInt (xmlParams, "ground_truth_queue_length");

	switch (params.targetOptimizationGroup) {
	case TARGET_POSITION:
		assert (typeid(TargetGroup) == typeid(Position) && "Need to recompile the project with using TargetGroup = lietorch::Position");
		break;
	case TARGET_QUATERNION_R4:
		assert (typeid(TargetGroup) == typeid(QuaternionR4) && "Need to recompile the project with using TargetGroup = lietorch::QuaternionR4");
		break;
	case TARGET_QUATERNION:
		assert (typeid(TargetGroup) == typeid(Quaternion) && "Need to recompile the project with using TargetGroup = lietorch::Quaternion");
		break;
	case TARGET_POSE_R4:
		assert (typeid(TargetGroup) == typeid(Pose3R4) && "Need to recompile the project with using TargetGroup = lietorch::Pose3R4");
		break;
	case TARGET_POSE:
		assert (typeid(TargetGroup) == typeid(Pose) && "Need to recompile the project with using TargetGroup = lietorch::Pose");
		break;
	case TARGET_DUAL_QUATERNION:
	default:
		assert (false && "Supplied target group id not supported");
		break;
	}

	return params;
}

typename PointcloudMatch<TargetGroup>::Params::Ptr
SlifeHandler::getCostFunctionParams (XmlRpc::XmlRpcValue &xmlParams)
{
	typename PointcloudMatch<TargetGroup>::Params::Ptr costFunctionParams = make_shared<PointcloudMatch<TargetGroup>::Params> ();

	costFunctionParams->batchSize = paramInt (xmlParams, "batch_size");
	costFunctionParams->stochastic = paramBool (xmlParams, "stochastic");
	costFunctionParams->reshuffleBatchIndexes = paramBool (xmlParams, "reshuffle_batch_indexes");

	return costFunctionParams;
}

typename PointcloudMatchOptimizer<TargetGroup>::Params::Ptr
SlifeHandler::getOptimizerParams (XmlRpc::XmlRpcValue &xmlParams)
{
	typename PointcloudMatchOptimizer<TargetGroup>::Params::Ptr optimizerParams = make_shared<PointcloudMatchOptimizer<TargetGroup>::Params> ();

	optimizerParams->stepSizes = paramTensor<float> (xmlParams, "step_sizes");
	optimizerParams->normWeights = paramTensor<float> (xmlParams, "norm_weights");
	optimizerParams->threshold = paramTensor<float> (xmlParams, "threshold");
	optimizerParams->maxIterations = paramTensor<float> (xmlParams, "max_iterations");
	optimizerParams->disable = paramBool (xmlParams, "disable");
	optimizerParams->initializationType = paramEnum<PointcloudMatchOptimizer<TargetGroup>::InitializationType> (xmlParams, "initialization_type",{"identity","last"});
	optimizerParams->recordHistory = paramBool (xmlParams, "record_history");

	return optimizerParams;
}

Landscape::Params::Ptr SlifeHandler::getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams)
{
	Landscape::Params::Ptr landscapeParams = make_shared<Landscape::Params> ();

	landscapeParams->measureRadius = paramDouble (xmlParams, "measure_radius");
	landscapeParams->smoothRadius = paramDouble (xmlParams, "smooth_radius");
	landscapeParams->precision = paramInt (xmlParams,"precision");
	landscapeParams->batchSize = paramInt (xmlParams, "batch_size");
	landscapeParams->maximumDistance = paramDouble (xmlParams, "maximum_distance");
	landscapeParams->decimation = paramInt (xmlParams,"decimation");
	landscapeParams->stochastic = paramBool (xmlParams, "stochastic");

	return landscapeParams;
}

GroundTruthTracker::GroundTruthTracker (const GroundTruthTracker::Params &_params):
	params(_params)
{}

void GroundTruthTracker::updateGroundTruth (const TargetGroup &groundTruth)
{
	groundTruths.push (groundTruth);

	if (groundTruths.size () > params.queueLength)
		groundTruths.pop ();
}

TargetGroup GroundTruthTracker::getRelativeGroundTruth () const {
	COUTN(groundTruths.front ());
	COUTN(groundTruths.back ());
	return groundTruths.front ().inverse() * groundTruths.back ();
}















