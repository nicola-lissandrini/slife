#ifndef SLIFE_HANDLER_H
#define SLIFE_HANDLER_H

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include "sparcsnode/common.h"
#include "optimizer.h"
#include "landscape.h"
#include "test.h"
#include "synchronization.h"


class OutputsManager;
extern std::vector<std::string> outputStrings;

class SlifeHandler
{
	using PointcloudWindow = ReadingWindow<Timed<Pointcloud>>;
	using TargetOptimizer = PointcloudMatchOptimizer<TargetGroup>;

public:
	enum TargetOptimizationGroup {
		TARGET_POSITION,
		TARGET_QUATERNION_R4,
		TARGET_QUATERNION,
		TARGET_POSE_R4,
		TARGET_POSE,
		TARGET_DUAL_QUATERNION
	};

	enum OutputType {
		OUTPUT_ESTIMATE,
		OUTPUT_HISTORY,
		OUTPUT_ERROR_HISTORY,
		OUTPUT_FINAL_ERROR,
		OUTPUT_RELATIVE_GROUND_TRUTH,
		OUTPUT_ESTIMATE_WORLD,
		OUTPUT_PROCESSED_POINTCLOUD,
		OUTPUT_MISC
	};

	struct Results {
		TargetGroup estimate;
		std::vector<TargetGroup> minima;
		TargetOptimizer::History history;
		std::vector<TargetOptimizer::History> histories;
		TargetGroup groundTruth;
		bool ready;

		Tensor historyTensor () const;
		Tensor historyErrorTensor () const;
		Tensor finalErrorTensor () const;

		Results ():
			ready(false)
		{}
	};

private:
	struct Params {
		bool syntheticPcl;
		bool normalizeBySampleTime;
		bool enableLocalMinHeuristics;
		OffsetEstimator::Params offsetEstimator;
		TargetOptimizationGroup targetOptimizationGroup;
		lietorch::Pose cameraFrame;
		PointcloudWindow::Params readingWindow;
		GroundTruthSync::Params groundTruthTracker;

		DEF_SHARED (Params)
	};

	Params params;
	ReadyFlags<std::string> flags;
	typename TargetOptimizer::Ptr optimizer;
	GroundTruthSync::Ptr groundTruthSync;
	std::shared_ptr<OutputsManager> outputsManager;
	PointcloudWindow::Ptr pointcloudWindow;

	FrequencyEstimator groundTruthFreq, pointcloudFreq;

	void outputResults (const Results &results);
	template<class LieGroup>
	LieGroup poseTensorToGroup (const torch::Tensor &poseTensor) const;
	TargetGroup normalizeToSampleTime(const TargetGroup &value, const FrequencyEstimator &frequency);
	template<class LieGroup>
	LieGroup cameraToGroundTruthFrame(const LieGroup &valueInCameraFrame);

	typename TargetOptimizer::Params::Ptr getOptimizerParams (XmlRpc::XmlRpcValue &xmlParams);
	typename PointcloudMatch<TargetGroup>::Params::Ptr getCostFunctionParams (XmlRpc::XmlRpcValue &xmlParams);

	Landscape::Params::Ptr getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams);
	SlifeHandler::Params getHandlerParams(XmlRpc::XmlRpcValue &xmlParams);

	Tensor miscTest (const Results &results);
	void test ();

public:
	SlifeHandler(const std::shared_ptr<OutputsManager> &_outputsManager);

	void init (XmlRpc::XmlRpcValue &xmlParams);
	void updatePointcloud (const Timed<Tensor> &timedPointcloud);
	void updateGroundTruth (const Timed<Tensor> &timedGroundTruthTensor);

	bool isSyntheticPcl() const;
	bool isReady () const;
	void pause ();
	void start ();
	void reset ();

	int synchronousActions ();

	DEF_SHARED (SlifeHandler)
};


#endif // SLIFE_HANDLER_H
