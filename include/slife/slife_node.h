#ifndef SLIFE_NODE_H
#define SLIFE_NODE_H

#include "sparcsnode/sparcsnode.h"
#include "slife/slife_handler.h"
#include "test.h"
#define NODE_NAME "slife"

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>

void transformToTensor (Tensor &out, const geometry_msgs::TransformStamped &transformMsg);

class SlifeNode : public SparcsNode
{
	SlifeHandler slifeHandler;
	geometry_msgs::TransformStamped lastGroundTruthMsg;
	torch::Tensor lastGroundTruthTensor;

	void initParams ();
	void initROS ();
	int actions ();

	void pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloudMsg);
	void groundTruthCallback (const geometry_msgs::TransformStamped &groundTruthMsg);
	void publishTensor (SlifeHandler::OutputTensorType outputType, const torch::Tensor &tensor);

public:
	SlifeNode ();

	friend class Test;

	DEF_SHARED(SlifeNode)
};



#endif // SLIFE_NODE_H
