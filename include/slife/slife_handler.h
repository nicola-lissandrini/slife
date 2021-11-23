#ifndef SLIFE_HANDLER_H
#define SLIFE_HANDLER_H

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include "common.h"
#include "optimizer.h"
#include "landscape.h"
#include "test.h"

class SlifeHandler
{
public:
	enum OutputTensorType {
		OUTPUT_ESTIMATE,
		OUTPUT_DEBUG_1,
		OUTPUT_DEBUG_2
	};

private:
	struct Params {
		int synthPclSize;
		DEF_SHARED (Params)
	};
	
	using TensorPublisher = std::function<void (OutputTensorType, const torch::Tensor &)>;

	Params params;
	ReadyFlags<std::string> flags;
	PointcloudMatch::Ptr costFunction;
	PoseOptimizer::Ptr optimizer;
	TensorPublisher tensorPublishCallback;

	Tensor historyToTensor (const std::vector<lietorch::Pose> &historyVector);

	PoseOptimizer::Params::Ptr getOptimizerParams (XmlRpc::XmlRpcValue &xmlParams);
	Landscape::Params::Ptr getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams);
	PointcloudMatch::Params::Ptr getCostFunctionParams(XmlRpc::XmlRpcValue &xmlParams);

	void test ();

public:
	SlifeHandler(TensorPublisher _tensorPublishCallback);

	void init (XmlRpc::XmlRpcValue &xmlParams);
	void updatePointcloud (const torch::Tensor &pointcloud);

	int synchronousActions ();
};

DEF_SHARED (SlifeHandler)

#endif // SLIFE_HANDLER_H
