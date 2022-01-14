#ifndef SLIFE_NODE_H
#define SLIFE_NODE_H

#include "sparcsnode/sparcsnode.h"
#include "slife/slife_handler.h"
#include "slife/Cmd.h"
#include "test.h"
#include <boost/optional.hpp>
#define NODE_NAME "slife"

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>

void transformToTensor (Tensor &out, const geometry_msgs::TransformStamped &transformMsg);

class SlifeNode : public SparcsNode
{
	SlifeHandler slifeHandler;
	ReadyFlagsStr readyFlags;
	ros::ServiceServer commandSrv;

	enum CmdOpCode {
		CMD_IS_READY = 0
	};

	void initParams ();
	void initROS ();
	int actions ();

	bool commandSrvCallback (slife::CmdRequest &request, slife::CmdResponse &response);
	void pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloudMsg);
	void groundTruthCallback (const geometry_msgs::TransformStamped &groundTruthMsg);
	void publishTensor (SlifeHandler::OutputTensorType outputType, const torch::Tensor &tensor, const std::vector<uint8_t> &extraData = std::vector<uint8_t> ());

public:
	SlifeNode ();

	friend class Test;

	DEF_SHARED(SlifeNode)
};



#endif // SLIFE_NODE_H
