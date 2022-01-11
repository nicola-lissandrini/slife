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

using TargetGroup = lietorch::Pose;

class GroundTruthTracker
{
public:
	struct Params {
		int queueLength;

		DEF_SHARED(Params)
	};

private:
	std::queue<TargetGroup> groundTruths;
	Params params;

public:
	GroundTruthTracker (const Params &_params);

	void updateGroundTruth (const TargetGroup &groundTruth);
	TargetGroup getRelativeGroundTruth () const;

	DEF_SHARED(GroundTruthTracker)
};

class SlifeHandler
{
public:
	enum OutputTensorType {
		OUTPUT_ESTIMATE,
		OUTPUT_DEBUG_1,
		OUTPUT_DEBUG_2
	};

	enum TargetOptimizationGroup {
		TARGET_POSITION,
		TARGET_QUATERNION_R4,
		TARGET_QUATERNION,
		TARGET_POSE_R4,
		TARGET_POSE,
		TARGET_DUAL_QUATERNION
	};

private:
	struct Params {
		bool syntheticPcl;
		TargetOptimizationGroup targetOptimizationGroup;
		GroundTruthTracker::Params groundTruthTracker;

		DEF_SHARED (Params)
	};
	
	using TensorPublisherExtra = std::function<void (OutputTensorType, const torch::Tensor &, const std::vector<uint8_t> &extraData)>;
	using TensorPublisher = std::function<void (OutputTensorType, const torch::Tensor &)>;

	Params params;
	ReadyFlags<std::string> flags;
	typename PointcloudMatchOptimizer<TargetGroup>::Ptr optimizer;
	GroundTruthTracker::Ptr groundTruthTracker;
	TensorPublisher tensorPublishCallback;
	TensorPublisherExtra tensorPublishExtraCallback;

	Tensor computeHistoryError (const std::vector<TargetGroup> &historyVector, const TargetGroup &groundTruth);
	Tensor historyToTensor (const std::vector<TargetGroup> &historyVector);
	template<class LieGroup>
	LieGroup poseTensorToGroup (const torch::Tensor &poseTensor) const;

	typename PointcloudMatchOptimizer<TargetGroup>::Params::Ptr getOptimizerParams (XmlRpc::XmlRpcValue &xmlParams);
	typename PointcloudMatch<TargetGroup>::Params::Ptr getCostFunctionParams (XmlRpc::XmlRpcValue &xmlParams);

	Landscape::Params::Ptr getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams);
	SlifeHandler::Params getHandlerParams(XmlRpc::XmlRpcValue &xmlParams);

	void test ();

public:
	SlifeHandler(const TensorPublisherExtra &_tensorPublisher);

	void init (XmlRpc::XmlRpcValue &xmlParams);
	void performOptimization (const torch::Tensor &pointcloud);
	void updateGroundTruth (const torch::Tensor &groundTruthTensor);
	bool isSyntheticPcl() const;

	int synchronousActions ();

	DEF_SHARED (SlifeHandler)
};


#endif // SLIFE_HANDLER_H
