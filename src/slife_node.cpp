#include "slife/slife_node.h"
#include "multi_array_manager.h"

#include <std_msgs/Empty.h>
#include <ATen/ATen.h>

#include <eigen3/Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/io/pcd_io.h>

#include <manif/SE2.h>

using namespace ros;
using namespace std;
using namespace torch;
using namespace Eigen;

Test::Ptr tester;

SlifeNode::SlifeNode ():
	SparcsNode(NODE_NAME),
	slifeHandler(std::bind (
				   &SlifeNode::publishTensor,
				   this,
				   std::placeholders::_1,
				   std::placeholders::_2))
{
	initParams ();
	initROS ();
}

void SlifeNode::initParams () {
	slifeHandler.init(params);
	tester = make_shared<Test> (params["debug"], shared_ptr<SlifeNode> (this));
}

void SlifeNode::initROS () {
	addSub ("pcl_sub", paramString (params["topics"], "pointcloud"), 2, &SlifeNode::pointcloudCallback);
	addPub<std_msgs::Float32MultiArray> ("test_range", paramString (params["topics"], "debug_grid"), 1);
	addPub<std_msgs::Float32MultiArray> ("estimate", paramString(params["topics"],"estimate"), 1);
	addPub<std_msgs::Float32MultiArray> ("debug_1", paramString(params["topics"], "debug_1"), 1);
	addPub<std_msgs::Float32MultiArray> ("debug_2", paramString(params["topics"], "debug_2"), 1);
}

int SlifeNode::actions ()  {
	return slifeHandler.synchronousActions();
}

void SlifeNode::publishTensor (SlifeHandler::OutputTensorType outputType, const Tensor &tensor)
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

// WARNING: takes 5-10ms
void SlifeNode::pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloud)
{
	const int pclSize = pointcloud.height * pointcloud.width;

	torch::Tensor pointcloudTensor;
	double taken;
	ROS_WARN ("Full Tensor loading");
	PROFILE (taken, [&]{
		pointcloudTensor = torch::from_blob ((void *) pointcloud.data.data(), {pclSize, 3}, // put this back for real pcl -> {pclSize, 4},
									  torch::TensorOptions().dtype(torch::kFloat32));
					    // REMOVING TEMPORARLY FOR SYNTHETIC PCL TESTS
					    // .index({torch::indexing::Ellipsis, torch::indexing::Slice(0,3)});
	});

	ROS_WARN ("Finding only finite points");
	PROFILE (taken,[&]{
		torch::Tensor validIdxes = (torch::isfinite(pointcloudTensor).sum(1)).nonzero();
		pointcloudTensor = pointcloudTensor.index ({validIdxes}).view ({validIdxes.size(0), D_3D});
	});

	slifeHandler.updatePointcloud(pointcloudTensor);
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

int main (int argc, char *argv[])
{
	init (argc, argv, NODE_NAME);

	SlifeNode sn;

	return sn.spin ();
}
