#include "slife/slife_handler.h"
#include "lietorch/algorithms.h"
#include "slife/slife_node.h"
#include <functional>

using namespace std;
using namespace torch;
using namespace lietorch;

SlifeHandler::SlifeHandler (const std::shared_ptr<OutputsManager> &_outputsManager):
	outputsManager(_outputsManager)
{
	flags.addFlag ("initialized", true);
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

void SlifeHandler::updateGroundTruth (const Timed<Tensor> &timedGroundTruthTensor)
{
	groundTruthFreq.tick (timedGroundTruthTensor.time ());
	Timed<TargetGroup> timedGroundTruth(timedGroundTruthTensor.time(),
								 poseTensorToGroup<TargetGroup> (timedGroundTruthTensor.obj()));
	groundTruthSync->updateGroundTruth (timedGroundTruth);
}

void SlifeHandler::updatePointcloud (const Timed<Tensor> &timedPointcloud)
{
	if (!groundTruthSync->groundTruthReady ())
		// Skip the first pointcloud if no ground truth received
		return;
	
	// Enqueue in window
	pointcloudWindow->add (timedPointcloud);
	
	if (!pointcloudWindow->isReady ())
		// Skip according to mode:
		// sliding: until queue full
		// decimate: every N packets
		return;
	
	Results currentResults;
	Timed<Pointcloud> currentTimedPointcloud = pointcloudWindow->get ();

	// Track decimated frequency
	pointcloudFreq.tick (currentTimedPointcloud.time ());
	
	// Track the ground truth corresponding to each pointcloud
	groundTruthSync->addSynchronizationMarker (currentTimedPointcloud.time ());

	// Set new pointcloud in cost function (landscape)
	optimizer->costFunction()->updatePointcloud (currentTimedPointcloud.obj ());

	// Do optimization, when ready
	if (optimizer->isReady() && groundTruthSync->markersReady ()) {
		if (params.enableLocalMinHeuristics) {
			optimizer->localMinHeuristics (currentResults.minima, currentResults.histories);
			currentResults.estimate = currentResults.minima[0];
			currentResults.history = currentResults.histories[0];
		} else
			optimizer->optimize(currentResults.estimate, currentResults.history);
		currentResults.ready = true;
	}
	
	currentResults.groundTruth = groundTruthSync->getRelativeGroundTruth ();
	
	outputResults (currentResults);
}

Tensor SlifeHandler::Results::finalErrorTensor () const {
	Tensor percentError = 100 * (groundTruth - estimate).norm ()/ groundTruth.log ().norm ();
	cout << "\nground truth: " << groundTruth << endl;
	cout << "estimate: " << estimate << endl;
	cout << "percent error: " << percentError.item ().toFloat () << "%\n" << endl;
	return (estimate.coeffs, groundTruth.coeffs);
}

TargetGroup SlifeHandler::normalizeToSampleTime (const TargetGroup &value, const FrequencyEstimator &frequency) {
	return  (value.log () * (1 / frequency.lastPeriodSeconds ())).exp ();
}

template<>
Pose SlifeHandler::cameraToGroundTruthFrame<Pose> (const Pose &valueInCameraFrame) {
	return params.cameraFrame * valueInCameraFrame;
}

template<>
Position SlifeHandler::cameraToGroundTruthFrame<Position> (const Position &valueInCameraFrame) {
	return params.cameraFrame.inverse () * valueInCameraFrame.coeffs;
}

Tensor SlifeHandler::miscTest (const Results &results)
{
	if (!results.ready)
		return Tensor ();

	for (auto curr : results.minima) {
		//COUTN(curr);
	}

	return Tensor ();
}

vector<string> outputStrings = {"estimate",
						  "history",
						  "error_history",
						  "final_error",
						  "relative_ground_truth",
						  "estimate_world",
						  "processed_pointcloud",
						  "misc"};

void SlifeHandler::outputResults (const Results &results)
{
	int i = 0;
	
	for (const OutputType &currType : outputsManager->getOutputs ()) {
		Tensor outputTensor;
		vector<float> extraData;
		
		if (currType != OUTPUT_RELATIVE_GROUND_TRUTH & currType != OUTPUT_MISC & !optimizer->isReady ())
			ROS_WARN_STREAM ("Cannot publish output '" << outputStrings[currType] << "': optimization is disabled or not ready");
		else {
			switch (currType) {
			case OUTPUT_ESTIMATE_WORLD:
			case OUTPUT_ESTIMATE:
				outputTensor = cameraToGroundTruthFrame (params.normalizeBySampleTime ?
													 normalizeToSampleTime (results.estimate,
																	    pointcloudFreq) :
													 results.estimate
												).coeffs;
				break;
			case OUTPUT_HISTORY:
				outputTensor = results.historyTensor ();
				extraData = {static_cast<float>(params.targetOptimizationGroup)};
				break;
			case OUTPUT_ERROR_HISTORY:
				outputTensor = results.historyErrorTensor ();
				extraData = {static_cast<float>(params.targetOptimizationGroup)};
				break;
			case OUTPUT_FINAL_ERROR:
				outputTensor = results.finalErrorTensor ();
				extraData = {static_cast<float>(params.targetOptimizationGroup)};
				break;
			case OUTPUT_RELATIVE_GROUND_TRUTH:
				outputTensor = params.normalizeBySampleTime ?
								normalizeToSampleTime (results.groundTruth,
												   pointcloudFreq).coeffs :
								results.groundTruth.coeffs;
				break;
			case OUTPUT_PROCESSED_POINTCLOUD:
				outputTensor = optimizer->costFunction ()->getPointcloud ();
				break;
			case OUTPUT_MISC:
				outputTensor = miscTest (results);
				break;
			}

			if (outputTensor.numel ()) {
				extraData.insert (extraData.begin (), currType);
				outputsManager->publishData (i, outputTensor, extraData);
			}
		}

		i++;
	}

}

Tensor SlifeHandler::Results::historyErrorTensor () const
{
	Tensor errorHistory = torch::empty ({(long int) history.size (),(long int) TargetGroup::Tangent::Dim}, kFloat);
	int i = 0;
	
	for (const TargetGroup &curr : history) {
		errorHistory[i] = (curr - groundTruth).coeffs;
		i++;
	}
	
	return errorHistory;
}

Tensor SlifeHandler::Results::historyTensor () const
{
	Tensor historiesTensor = torch::empty({(long int) histories.size (), (long int) history.size (), TargetGroup::Dim}, kFloat);
	int i = 0;

	for (auto currHistory : histories) {
		int j = 0;

		for (auto currEstimate : currHistory) {
			historiesTensor[i][j] = currEstimate.coeffs;
			j++;
		}
		i++;
	}
	
	return historiesTensor;
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

	groundTruthSync = make_shared<GroundTruthSync> (params.groundTruthTracker);
	pointcloudWindow = make_shared<PointcloudWindow> (params.readingWindow);
	
	flags.set ("initialized");
}

bool SlifeHandler::isSyntheticPcl () const {
	return params.syntheticPcl;
}

bool SlifeHandler::isReady() const {
	return flags.isReady ();
}

void SlifeHandler::pause() {
	optimizer->disable ();
}

void SlifeHandler::start () {
	optimizer->enable ();
}

void SlifeHandler::reset()
{
	pointcloudFreq.reset ();
	groundTruthFreq.reset ();
	groundTruthSync.reset ();
	optimizer->reset ();
}

SlifeHandler::Params SlifeHandler::getHandlerParams (XmlRpc::XmlRpcValue &xmlParams)
{
	Params params;
	
	params.syntheticPcl = paramBool (xmlParams, "synthetic_pcl");
	params.targetOptimizationGroup = paramEnum<TargetOptimizationGroup> (xmlParams, "target_optimization_group",{"position","quaternion_r4","quaternion","pose_r4","pose","dual_quaternion"});
	params.cameraFrame.coeffs = paramTensor<float> (xmlParams, "vicon_to_camera_frame");
	params.readingWindow.mode = paramEnum<PointcloudWindow::Mode> (xmlParams, "window_mode", {"sliding", "downsample"});
	params.readingWindow.size = paramInt (xmlParams, "window_size");
	params.groundTruthTracker.queueLength = paramInt (xmlParams["ground_truth_sync"], "queue_length");
	params.groundTruthTracker.msOffset = paramFloat (xmlParams["ground_truth_sync"], "ms_offset");
	params.normalizeBySampleTime = paramBool (xmlParams, "normalize_by_sample_time");
	params.enableLocalMinHeuristics = paramBool (xmlParams, "enable_local_min_heuristics");
	params.offsetEstimator.activationThreshold = paramFloat (xmlParams["offset_estimator"], "activation_threshold");
	params.offsetEstimator.enable = paramFloat (xmlParams["offset_estimator"], "enable");
	
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
	optimizerParams->localMinHeuristics.count = paramInt (xmlParams["local_min_heuristics"], "count");
	optimizerParams->localMinHeuristics.scatter = paramFloat (xmlParams["local_min_heuristics"], "scatter");
	
	return optimizerParams;
}

Landscape::Params::Ptr SlifeHandler::getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams)
{
	Landscape::Params::Ptr landscapeParams = make_shared<Landscape::Params> ();
	
	landscapeParams->measureRadius = paramDouble (xmlParams, "measure_radius");
	landscapeParams->smoothRadius = paramDouble (xmlParams, "smooth_radius");
	landscapeParams->precision = paramInt (xmlParams,"precision");
	landscapeParams->batchSize = paramInt (xmlParams, "batch_size");
	landscapeParams->clipArea = paramRange (xmlParams, "clip_area");
	landscapeParams->decimation = paramInt (xmlParams,"decimation");
	landscapeParams->stochastic = paramBool (xmlParams, "stochastic");
	
	return landscapeParams;
}















