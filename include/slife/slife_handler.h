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
		TARGET_POSE,
		TARGET_DUAL_QUATERNION
	};

private:
	struct Params {
		int synthPclSize;
		TargetOptimizationGroup targetOptimizationGroup;
		DEF_SHARED (Params)
	};
	
	using TensorPublisher = std::function<void (OutputTensorType, const torch::Tensor &)>;

	Params params;
	ReadyFlags<std::string> flags;
	typename PointcloudMatchOptimizer<TargetGroup>::Ptr optimizer;
	TensorPublisher tensorPublishCallback;

	Tensor historyToTensor (const std::vector<TargetGroup> &historyVector);

	typename PointcloudMatchOptimizer<TargetGroup>::Params::Ptr getOptimizerParams (XmlRpc::XmlRpcValue &xmlParams);
	typename PointcloudMatch<TargetGroup>::Params::Ptr getCostFunctionParams (XmlRpc::XmlRpcValue &xmlParams);

	Landscape::Params::Ptr getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams);
	SlifeHandler::Params getHandlerParams(XmlRpc::XmlRpcValue &xmlParams);

	void test ();

public:
	SlifeHandler(const TensorPublisher &_tensorPublisher);

	void init (XmlRpc::XmlRpcValue &xmlParams);
	void updatePointcloud (const torch::Tensor &pointcloud);

	int synchronousActions ();

	DEF_SHARED (SlifeHandler)
};


#endif // SLIFE_HANDLER_H
