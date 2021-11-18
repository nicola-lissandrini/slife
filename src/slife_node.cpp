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

using namespace ros;
using namespace std;
using namespace torch;
using namespace Eigen;

Test::Ptr tester;

SlifeNode::SlifeNode ():
	SparcsNode(NODE_NAME)
{
	initParams ();
	initROS ();
}

void SlifeNode::initParams () {
	slifeHandler.init(params);
	tester = make_shared<Test> (params["debug"], shared_ptr<SlifeNode> (this));
}

void SlifeNode::initROS () {
	addSub ("pcl_sub", paramString (params["topics"], "pointcloud"), 1, &SlifeNode::pointcloudCallback);
	addPub<std_msgs::Float32MultiArray> ("test_range", paramString (params["topics"], "debug_grid"), 1);
}

int SlifeNode::actions ()  {
	return slifeHandler.synchronousActions();
}


void SlifeNode::pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloud)
{
	QUA;
	pcl::PCLPointCloud2Ptr pcl2(new pcl::PCLPointCloud2), reduced(new pcl::PCLPointCloud2);
	pcl::PointCloud<pcl::PointXYZ> pcl3;
	float taken;
	cout << "type conversion " << endl;
	PROFILE(taken,[&]{
		pcl_conversions::toPCL(pointcloud, *pcl2);
	});

	pcl::fromPCLPointCloud2 (*pcl2, pcl3);
	pcl::NarfKeypoint  *narfKeypointDetector;

	PROFILE(taken,[&]{
		 float noise_level = 0.0;
		 float min_range = 0.0f;
		 int border_size = 1;
		 pcl::RangeImage::Ptr rangeImagePtr(new pcl::RangeImage);
		 pcl::RangeImage &rangeImage = *rangeImagePtr;

		 rangeImage.createFromPointCloud (pcl3, pcl::deg2rad (0.5f), pcl::deg2rad (360.0f),
								    pcl::deg2rad (180.0f),
								    Eigen::Affine3f::Identity ());
		 rangeImage.setUnseenToMaxRange ();
		 pcl::RangeImageBorderExtractor rangeImageBorderExtractor;
		 narfKeypointDetector = new pcl::NarfKeypoint (&rangeImageBorderExtractor);

		 narfKeypointDetector->setRangeImage (&rangeImage);
		 narfKeypointDetector->getParameters().support_size = 0.2f;
		 pcl::PointCloud<int> keypointIndices;
		 narfKeypointDetector->compute (keypointIndices);

		 std::cout << "Found "<<keypointIndices.size ()<<" key points.\n" << endl;
	});


	cout << reduced->width << "x" << reduced->height << endl;
	QUA;
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
	params.testType = paramEnum<Type> (xmlParams, "test_type", {"none","value","gradient"});
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
