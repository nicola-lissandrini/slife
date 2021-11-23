#include "slife/slife_handler.h"

#include <octomap/OcTree.h>

using namespace std;
using namespace torch;
using namespace lietorch;

SlifeHandler::SlifeHandler(TensorPublisher _tensorPublishCallback):
	tensorPublishCallback(_tensorPublishCallback)
{
	flags.addFlag ("initialized", true);
}

void SlifeHandler::updatePointcloud (const Tensor &pointcloud) {
	costFunction->updatePointcloud(pointcloud);

	if (costFunction->isReady()) {
		lietorch::Pose estimate = optimizer->optimize();

		tensorPublishCallback (OUTPUT_ESTIMATE, estimate.coeffs);
		tensorPublishCallback (OUTPUT_DEBUG_1, historyToTensor (optimizer->getHistory ()));
	}
}

Tensor SlifeHandler::historyToTensor (const std::vector<Pose> &historyVector)
{
	Tensor historyTensor = torch::empty({historyVector.size(), Pose::Dim}, kFloat);
	int i = 0;

	for (Pose curr : historyVector) {
		historyTensor[i] = curr.coeffs;
		i++;
	}

	return historyTensor;
}

void SlifeHandler::test ()
{
	Test::Type testWhat = tester->getType();
	Tensor testValues;

	testValues = costFunction->test (testWhat);

	if (testValues.numel ())
		tester->publishRangeTensor (testWhat, testValues);
}

int SlifeHandler::synchronousActions ()
{
	/*if (costFunction->isReady())
		test ();*/

	return 0;
}

void SlifeHandler::init (XmlRpc::XmlRpcValue &xmlParams)
{
	Landscape::Params::Ptr landscapeParams = getLandscapeParams(xmlParams["landscape"]);
	PoseOptimizer::Params::Ptr optimizerParams = getOptimizerParams(xmlParams["optimizer"]);
	PointcloudMatch::Params::Ptr costFunctionParams = getCostFunctionParams(xmlParams["optimizer"]["cost"]);

	costFunction = make_shared<PointcloudMatch> (landscapeParams,
										costFunctionParams);
	optimizer = make_shared<PoseOptimizer> (optimizerParams,
									dynamic_pointer_cast<CostFunction<lietorch::Pose>> (costFunction));

	params.synthPclSize = paramInt (xmlParams, "synth_pcl_size");

	flags.set ("initialized");
}

PointcloudMatch::Params::Ptr SlifeHandler::getCostFunctionParams (XmlRpc::XmlRpcValue &xmlParams)
{
	PointcloudMatch::Params::Ptr costFunctionParams = make_shared<PointcloudMatch::Params> ();

	costFunctionParams->batchSize = paramInt (xmlParams, "batch_size");

	return costFunctionParams;
}

PoseOptimizer::Params::Ptr SlifeHandler::getOptimizerParams (XmlRpc::XmlRpcValue &xmlParams)
{
	PoseOptimizer::Params::Ptr optimizerParams = make_shared<PoseOptimizer::Params> ();

	optimizerParams->stepSizes = paramTensor<float> (xmlParams, "step_sizes");
	optimizerParams->threshold = paramTensor<float> (xmlParams, "threshold");
	optimizerParams->maxIterations = paramTensor<float> (xmlParams, "max_iterations");
	optimizerParams->initializationType = paramEnum<PoseOptimizer::InitializationType> (xmlParams, "initialization_type",{"identity"});
	optimizerParams->recordHistory = paramBool (xmlParams, "record_history");

	return optimizerParams;
}

Landscape::Params::Ptr SlifeHandler::getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams)
{
	Landscape::Params::Ptr landscapeParams = make_shared<Landscape::Params> ();

	landscapeParams->measureRadius = paramDouble (xmlParams, "measure_radius");
	landscapeParams->smoothRadius = paramDouble (xmlParams, "smooth_radius");
	landscapeParams->precision = paramInt (xmlParams,"precision");

	return landscapeParams;
}













