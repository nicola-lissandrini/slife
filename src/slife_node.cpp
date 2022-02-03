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
	outputsManager(&nh),
	slifeHandler(shared_ptr<OutputsManager> (&outputsManager))
{
	initParams ();
	initROS ();
	outputsManager.init (params);

	readyFlags.addFlag ("ready_sent");
}

void SlifeNode::initParams () {
	tester = make_shared<Test> (params["debug"], shared_ptr<SlifeNode> (this));
	slifeHandler.init(params);
}

void SlifeNode::initROS ()
{
	addSub ("pcl_sub", paramString (params["topics"], "pointcloud"), 2000, &SlifeNode::pointcloudCallback);
	addSub ("ground_truth_sub", paramString (params["topics"], "ground_truth"), 2000, &SlifeNode::groundTruthCallback);

	addPub<std_msgs::Float32MultiArray> ("test_range", paramString (params["topics"], "debug_grid"), 1);
	addPub<std_msgs::Float32MultiArray> ("estimate", paramString(params["topics"],"estimate"), 1);

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
	case CMD_PAUSE:
		if (slifeHandler.isReady ()) {
			slifeHandler.pause ();
			response.response = 1;
		} else
			response.response = 0;
		break;
	case CMD_START:
		slifeHandler.start ();
		response.response = 1;
		break;
	default:
		ROS_WARN ("Unrecognized command received %d", request.command);
		return false;
	}

	return true;
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

OutputsManager::OutputsManager (NodeHandle *_nh):
	nh(_nh)
{
}

void OutputsManager::init (XmlRpc::XmlRpcValue &params)
{
	string topicPrefix = paramString (params["topics"], "output_prefix");

	outputs.push_back (SlifeHandler::OUTPUT_ESTIMATE);

	const vector<SlifeHandler::OutputType> &optionalOutputs =
			paramArray<SlifeHandler::OutputType> (params, "outputs",
										   [] (XmlRpc::XmlRpcValue &param) {
		return paramEnum<SlifeHandler::OutputType> (param, outputStrings);
	});

	outputs.insert (outputs.end(), optionalOutputs.begin (), optionalOutputs.end ());

	pubs.push_back (nh->advertise<std_msgs::Float32MultiArray> (paramString (params["topics"], "estimate"), 1));

	for (int i = 1; i < outputs.size (); i++) {
		string topic = topicPrefix + "/output_" + to_string (i);
		if (outputs[i] == SlifeHandler::OUTPUT_PROCESSED_POINTCLOUD)
			pubs.push_back (nh->advertise<sensor_msgs::PointCloud2> (topic, 1));
		else
			pubs.push_back (nh->advertise<std_msgs::Float32MultiArray> (topic, 1));
	}

}

const std::vector<SlifeHandler::OutputType> &OutputsManager::getOutputs() const {
	return outputs;
}

void OutputsManager::tensorToMsg (std_msgs::Float32MultiArray &outputMsg, const torch::Tensor &tensor, const std::vector<float> &extraData)
{
	MultiArray32Manager array(vector<int> (tensor.sizes().begin(), tensor.sizes().end()), extraData.size ());

	memcpy (array.data ().data(), extraData.data (), extraData.size () * sizeof (float));
	memcpy (((float *) array.data ().data()) + extraData.size (), tensor.data_ptr(), tensor.element_size() * tensor.numel ());

	outputMsg = array.getMsg();
}

void OutputsManager::tensorToMsg (sensor_msgs::PointCloud2 &outputMsg, const torch::Tensor &tensor, const std::vector<float> &extraData)
{
	sensor_msgs::PointField pointFieldProto;
	pointFieldProto.datatype = 7;
	pointFieldProto.count = 1;

	pointFieldProto.name = "x";
	pointFieldProto.offset = 0;
	outputMsg.fields.push_back (pointFieldProto);
	pointFieldProto.name = "y";
	pointFieldProto.offset = 4;
	outputMsg.fields.push_back (pointFieldProto);
	pointFieldProto.name = "z";
	pointFieldProto.offset = 8;
	outputMsg.fields.push_back (pointFieldProto);

	outputMsg.height = 1;
	outputMsg.width = tensor.size(0);

	outputMsg.data = vector<uint8_t> ((uint8_t*)tensor.data_ptr (), (uint8_t*)tensor.data_ptr () + tensor.numel () * tensor.element_size ());
	outputMsg.header.frame_id = "map";
	outputMsg.point_step = D_3D * tensor.element_size ();
}


void OutputsManager::publishData (int outputId, const Tensor &tensor, const std::vector<float> &extraData)
{
	if (outputs[outputId] == SlifeHandler::OUTPUT_PROCESSED_POINTCLOUD) {
		sensor_msgs::PointCloud2 pointcloudMsg;
		tensorToMsg (pointcloudMsg, tensor, extraData);
		pubs[outputId].publish (pointcloudMsg);
	} else {
		std_msgs::Float32MultiArray tensorMsg;
		tensorToMsg (tensorMsg, tensor, extraData);
		pubs[outputId].publish (tensorMsg);
	}
}


void handler(int sig)  {
	STACKTRACE;
	exit(-1);
}

int main (int argc, char *argv[])
{
	signal(SIGSEGV, handler);
	signal(SIGABRT, handler);
	init (argc, argv, NODE_NAME);

	SlifeNode sn;

	return sn.spin ();
}

























