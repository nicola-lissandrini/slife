#include "slife/slife_node.h"
#include "multi_array_manager.h"

#include <std_msgs/Empty.h>
#include <ATen/ATen.h>

#include <eigen3/Eigen/Core>

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

// WARNING: takes 5-10ms
void SlifeNode::pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloud)
{
	const int pclSize = pointcloud.height * pointcloud.width;

	torch::Tensor pointcloudTensor;

	double taken;
	ROS_WARN ("Full Tensor loading");
	PROFILE (taken, [&]{
		pointcloudTensor = torch::from_blob ((void *) pointcloud.data.data(), {pclSize, 4},
									  torch::TensorOptions().dtype(torch::kFloat32))
					    .index({torch::indexing::Ellipsis, torch::indexing::Slice(0,3)});
	});
	ROS_WARN ("Finding only finite points");

	PROFILE (taken,[&]{
		/*torch::Tensor validPointcloud = torch::empty_like(pointcloudTensor);
		int j = 0;
		for (int i = 0; i < pointcloudTensor.size(0); i++) {
			cout << pointcloudTensor[i].isfinite();
			/*if (pointcloudTensor[i].isfinite().sum(0)) {
				validPointcloud[j] = pointcloudTensor[i];
				j++;
			}
		}
		pointcloudTensor = validPointcloud.index ({torch::indexing::Slice(0,j),torch::indexing::Ellipsis});*/
		torch::Tensor validIdxes = (torch::isfinite(pointcloudTensor).sum(1)).nonzero();
		pointcloudTensor = pointcloudTensor.index ({validIdxes}).view ({validIdxes.size(0), D_3D});
	});



	PROFILE (taken,[&]{
		torch::Tensor validIdxes = (torch::isfinite(pointcloudTensor.sum(1))).nonzero();
		pointcloudTensor = pointcloudTensor.index ({validIdxes}).view ({validIdxes.size(0), D_3D});
	});

	torch::Tensor decimatedPcl;

	ROS_WARN ("Decimate pcl");
	PROFILE (taken, [&] {
		 decimatedPcl = pointcloudTensor.slice (0, 0, c10::nullopt, 10);
	});

	ROS_WARN ("Nan in decimated pcl");
	PROFILE (taken,[&]{
		torch::Tensor validIdxes = (torch::isfinite(decimatedPcl).sum(1)).nonzero();
		decimatedPcl = decimatedPcl.index ({validIdxes}).view ({validIdxes.size(0), D_3D});
	});




	/*ROS_WARN ("alg 2");
	PROFILE (taken, [&]{
		Tensor validPcl = torch::empty_like(pointcloudTensor);
		int j = 0;
		for (int i = 0; i < validPcl.size(0); i++) {
			if ((pointcloudTensor[i].sum() > 0).item().toBool()) {
				validPcl[j] = pointcloudTensor[i];
				j++;
			}
		}
		validPcl = validPcl.slice (0, j);
	});*/

	//Matrix<float, 407040, 3> pclEigen;
	//Matrix<float, 407040, 3> validPclEigen;
	using PclEigen = Matrix<float, 407040, 3>;

	cout << "loading eigen" << endl;
	PclEigen pclEigen;
	PROFILE (taken, [&]{
		  pclEigen = Map<PclEigen> ((float*)pointcloud.data.data());
	});

	ROS_WARN ("x == x");
	PROFILE (taken, [&]{
		auto x = (pclEigen - pclEigen).array();
		cout << "before" << endl;
		cout << abi::__cxa_demangle(typeid(x).name(),0,0,0) << endl;
		cout << x << endl;
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

	testGrid.points = stack ({xx, yy,  params.zTestValue * torch::ones_like(xx)}, 1);
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
