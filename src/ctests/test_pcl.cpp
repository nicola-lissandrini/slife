#include <iostream>
#include <Eigen/Core>
#include <torch/all.h>
#include <cxxabi.h>

#include "../../sparcslib/include/profiling.h"

using namespace torch;
using namespace std;
using namespace Eigen;

#define PCL_N 407040

using PclEigenEx = Matrix<float, PCL_N, 4, RowMajor>;
using PclEigen = Matrix<float, PCL_N, 3, RowMajor>;

int main ()
{
	float taken;
	Tensor pcl, summed;
	PclEigen pclEigen;

	cout << "aaA" << endl;
	torch::load (pcl, "/home/nicola/pcl.torch");

	cout << "aaA" << endl;
	Map<PclEigenEx> mp((float*)pcl.data_ptr());

	cout << "aaA" << endl;

	Array<float, PCL_N, 1> c;
	Array<bool, PCL_N, 1> e;

	for (int i = 0; i < 10; i++) {
		cout <<  endl << "eigen sum rowwise" << endl;
		PROFILE (taken,[&]{
			c = mp.rowwise().sum().array();
			c =c( (c - c) != (c - c));
		});

		cout << "torch sum rowwise" << endl;
		PROFILE (taken,[&]{
			summed = pcl.sum(1).isnan();
		});
	}

	cout << "eig tail 10\n" <<  e.bottomRows<10> () << endl;
	cout << "torch tail 10\n" <<  summed.slice(0,PCL_N-10, PCL_N) << endl;


	cout << "nan pointcloud selection" << endl;



	return 0;
}














