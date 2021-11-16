#include <iostream>
#include <torch/all.h>
#include <cxxabi.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/conversions.h>

#include "../../sparcslib/include/profiling.h"

using namespace torch;
using namespace std;
using namespace pcl;

#define PCL_N 407040

float flt(const torch::Tensor &t) {
	return t.item().toFloat();
}

int main ()
{
	float taken;
	Tensor pcld, summed;
	sensor_msgs::PointCloud2 pclMsg;
	PCLPointCloud2::Ptr pcl1(new PCLPointCloud2);

	torch::load (pcld, "/home/nicola/pcl.torch");


	pcl1->width = PCL_N;
	pcl1->height = 1;
	pcl1->data = vector<uint8_t> ((uint8_t*) pcld.data_ptr(), (uint8_t*) pcld.data_ptr() + pcld.numel()*sizeof(float));
	pcl1->fields = {
		{"x", 0, 7, 1},
		{"y", 0, 7, 1},
		{"z", 0, 7, 1},
	};

	VoxelGrid<PCLPointCloud2> sor;

	sor.setInputCloud(pcl1);
	sor.setLeafSize(100.01,100.01,100.01);
	sor.filter(*pcl2);

	cout << "after filtering\n" << pcl2->data.size() << endl;

	return 0;
}














