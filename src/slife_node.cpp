#include "slife/slife_node.h"
#include "../../sparcsnode/include/sparcsnode/multi_array_manager.h"
#include <boost/date_time/posix_time/conversion.hpp>

#include <std_msgs/Empty.h>
#include <ATen/ATen.h>

using namespace ros;
using namespace std;
using namespace torch;


Test::Ptr tester;

chrono::time_point<chrono::system_clock> rosTimeToStd (const ros::Time &rosTime) {
	return chrono::time_point<chrono::system_clock> () + chrono::nanoseconds(rosTime.toNSec ());
}

SlifeNode::SlifeNode ():
	SparcsNode(NODE_NAME),
	slifeHandler([this](SlifeHandler::OutputTensorType outputType,
					const torch::Tensor &tensor,
					const std::vector<uint8_t> &extraData)
			   { return publishTensor (outputType, tensor, extraData);})
{
	initParams ();
	initROS ();

	readyFlags.addFlag ("ready_sent");
}

void SlifeNode::initParams () {
	tester = make_shared<Test> (params["debug"], shared_ptr<SlifeNode> (this));
	slifeHandler.init(params);
}

void SlifeNode::initROS () {
	addSub ("pcl_sub", paramString (params["topics"], "pointcloud"), 2, &SlifeNode::pointcloudCallback);
	addSub ("ground_truth_sub", paramString (params["topics"], "ground_truth"), 2, &SlifeNode::groundTruthCallback);

	addPub<std_msgs::Float32MultiArray> ("test_range", paramString (params["topics"], "debug_grid"), 1);
	addPub<std_msgs::Float32MultiArray> ("estimate", paramString(params["topics"],"estimate"), 1);
	addPub<std_msgs::Float32MultiArray> ("debug_1", paramString(params["topics"], "debug_1"), 1);
	addPub<std_msgs::Float32MultiArray> ("debug_2", paramString(params["topics"], "debug_2"), 1);

	commandSrv = nh.advertiseService (paramString (params["topics"], "command"), &SlifeNode::commandSrvCallback, this);
}

int SlifeNode::actions ()  {
	return slifeHandler.synchronousActions();
}

bool SlifeNode::commandSrvCallback (slife::CmdRequest &request, slife::CmdResponse &response)
{
	CmdOpCode cmd = (CmdOpCode) request.command;

	switch (cmd) {
	case CMD_IS_READY:
		response.response = (int64_t) slifeHandler.isReady ();
		break;
	default:
		ROS_WARN ("Unrecognized command received %d", request.command);
		return false;
	}

	return true;
}

void SlifeNode::publishTensor (SlifeHandler::OutputTensorType outputType, const Tensor &tensor, const std::vector<uint8_t> &extraData)
{
	MultiArray32Manager array(vector<int> (tensor.sizes().begin(), tensor.sizes().end()));
	
	memcpy (array.data ().data(), tensor.data_ptr(), tensor.element_size() * tensor.numel ());
	auto tensorMsg = array.getMsg();
	
	switch (outputType) {
	case SlifeHandler::OUTPUT_ESTIMATE:
		publish ("estimate", tensorMsg);
		break;
	case SlifeHandler::OUTPUT_DEBUG_1:
		publish ("debug_1", tensorMsg);
		break;
	case SlifeHandler::OUTPUT_DEBUG_2:
		publish ("debug_2", tensorMsg);
		break;
	}
}

void SlifeNode::pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloudMsg)
{
	const int pclSize = pointcloudMsg.height * pointcloudMsg.width;
	Tensor pointcloudTensor;
	Timed<Tensor> timedPointcloud;

	if (slifeHandler.isSyntheticPcl ()) {
		pointcloudTensor = torch::from_blob ((void *) pointcloudMsg.data.data(), {pclSize, 3},
									  torch::TensorOptions().dtype(torch::kFloat32));
	} else {
		pointcloudTensor = torch::from_blob ((void *) pointcloudMsg.data.data(), {pclSize, 4},
									  torch::TensorOptions().dtype (torch::kFloat32))
					    .index ({indexing::Ellipsis, indexing::Slice(0,3)});
	}
	
	timedPointcloud.obj () = pointcloudTensor;

	timedPointcloud.time () = rosTimeToStd (pointcloudMsg.header.stamp);

	slifeHandler.updatePointcloud(timedPointcloud);
}

void SlifeNode::groundTruthCallback (const geometry_msgs::TransformStamped &groundTruthMsg)
{
	Timed<Tensor> timedGroundTruthTensor;
	transformToTensor (timedGroundTruthTensor.obj (), groundTruthMsg);
	timedGroundTruthTensor.time () = rosTimeToStd (groundTruthMsg.header.stamp);

	slifeHandler.updateGroundTruth (timedGroundTruthTensor);
}

void transformToTensor (Tensor &out, const geometry_msgs::TransformStamped &transformMsg)
{
	out = torch::tensor ({transformMsg.transform.translation.x,
					  transformMsg.transform.translation.y,
					  transformMsg.transform.translation.z,
					  transformMsg.transform.rotation.x,
					  transformMsg.transform.rotation.y,
					  transformMsg.transform.rotation.z,
					  transformMsg.transform.rotation.w}, kFloat);
}

Tensor Test::getTestGrid() const {
	return testGrid.points;
}

int Test::getTestGridSize() const {
	return testGrid.xySize;
}

Test::Type Test::getType() const {
	return params.testType;
}

void Test::initTestGrid()
{
	Tensor xyRange = torch::arange (params.testGridRanges.min, params.testGridRanges.max, params.testGridRanges.step, torch::dtype (kFloat));
	Tensor xx, yy;
	vector<Tensor> xy;
	
	xy = meshgrid ({xyRange, xyRange});
	
	xx = xy[0].reshape (-1);
	yy = xy[1].reshape (-1);
	
	testGrid.points = torch::stack ({xx, yy,  params.zTestValue * torch::ones_like(xx)}, 1);
	testGrid.xySize = xyRange.size (0);
}

#define TEST_HEADER_SIZE 4

void Test::publishRangeTensor(Test::Type type, const Tensor &tensor)
{
	MultiArray32Manager array(vector<int> (tensor.sizes().begin(), tensor.sizes().end()), TEST_HEADER_SIZE);
	
	array.data()[0] = params.testGridRanges.min;
	array.data()[1] = params.testGridRanges.max;
	array.data()[2] = params.testGridRanges.step;
	array.data()[3] = (float) type;
	
	memcpy (array.data().data() + TEST_HEADER_SIZE,tensor.data_ptr(), tensor.element_size()*tensor.numel());
	
	auto valuesMsg = array.getMsg();
	
	nodePtr->publish ("test_range", valuesMsg);
}

void Test::initParams (XmlRpc::XmlRpcValue &xmlParams)
{
	params.testGridRanges = paramRange (xmlParams,"test_grid");
	params.zTestValue = paramDouble (xmlParams, "z_test_value");
	params.testType = paramEnum<Type> (xmlParams, "test_type", {"none","landscape_value", "landscape_gradient", "cost_value", "cost_gradient"});
}

Test::Test (XmlRpc::XmlRpcValue &xmlParams,
		  const std::shared_ptr<SlifeNode> &_nodePtr):
	nodePtr(_nodePtr)
{
	initParams (xmlParams);
	initTestGrid ();
}

void handler(int sig)  {
	STACKTRACE;
	exit(-1);
}

int main (int argc, char *argv[])
{
	signal(SIGSEGV, handler);
	init (argc, argv, NODE_NAME);

	SlifeNode sn;

	return sn.spin ();
}
