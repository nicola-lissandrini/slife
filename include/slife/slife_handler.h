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
	struct Params {
		int synthPclSize;
		DEF_SHARED (Params)
	};

	Params params;
	ReadyFlags<std::string> flags;
	PointcloudMatch::Ptr costFunction;
	PoseOptimizer::Ptr optimizer;

	PoseOptimizer::Params::Ptr getOptimizerParams (XmlRpc::XmlRpcValue &xmlParams);
	Landscape::Params::Ptr getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams);
	PointcloudMatch::Params::Ptr getCostFunctionParams(XmlRpc::XmlRpcValue &xmlParams);

	void test ();

public:
	SlifeHandler();

	void init (XmlRpc::XmlRpcValue &xmlParams);
	void updatePointcloud (const torch::Tensor &pointcloud);

	int synchronousActions ();
};

DEF_SHARED (SlifeHandler)

#endif // SLIFE_HANDLER_H
