#ifndef SLIFE_NODE_H
#define SLIFE_NODE_H

#include "sparcsnode/sparcsnode.h"
#include "slife/slife_handler.h"
#include "slife/Cmd.h"
#include "test.h"
#include <boost/optional.hpp>
#define NODE_NAME "slife"

#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>

void transformToTensor (Tensor &out, const geometry_msgs::TransformStamped &transformMsg);

class OutputsManager
{
	ros::NodeHandle *nh;
	std::vector<SlifeHandler::OutputType> outputs;
	std::vector<ros::Publisher> pubs;

	void tensorToMsg (std_msgs::Float32MultiArray &outputMsg, const torch::Tensor &tensor, const std::vector<float> &extraData = std::vector<float> ());
	void tensorToMsg (sensor_msgs::PointCloud2 &outputMsg, const torch::Tensor &tensor, const std::vector<float> &extraData = std::vector<float> ());


public:
	OutputsManager (ros::NodeHandle *nh);

	void init (XmlRpc::XmlRpcValue &params);
	void publishData (int outputId, const torch::Tensor &tensor, const std::vector<float> &extraData = std::vector<float> ());
	const std::vector<SlifeHandler::OutputType> &getOutputs () const;

	DEF_SHARED(OutputsManager)
};

class SlifeNode : public SparcsNode
{
	OutputsManager outputsManager;
	SlifeHandler slifeHandler;
	ReadyFlagsStr readyFlags;
	ros::ServiceServer commandSrv;

	enum CmdOpCode {
		CMD_IS_READY = 0,
		CMD_PAUSE,
		CMD_START
	};

	void initParams ();
	void initROS ();
	int actions ();

	bool commandSrvCallback (slife::CmdRequest &request, slife::CmdResponse &response);
	void pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloudMsg);
	void groundTruthCallback (const geometry_msgs::TransformStamped &groundTruthMsg);

public:
	SlifeNode ();

	friend class Test;

	DEF_SHARED(SlifeNode)
};



#endif // SLIFE_NODE_H
