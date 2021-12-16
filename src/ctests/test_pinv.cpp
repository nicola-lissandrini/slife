#include <torch/all.h>
#include <iostream>

#include "../../sparcsnode/include/sparcsnode/utils.h"
#include "../../sparcsnode/include/sparcsnode/profiling.h"


using namespace torch;
using namespace std;

#define N 1000

int main ()
{
	double total1 = 0, total2 = 0;

	double taken1, taken2;
	for (int i = 0; i < N; i++) {
		Tensor jacobian = torch::rand ({3, 2000}, kFloat);

		Tensor p1, p2;
		PROFILE_N_EN (taken1, [&]{
			p1 = (jacobian.mm(jacobian.t())).inverse().mm (jacobian);
		},1,false);


		PROFILE_N_EN (taken2, [&]{
			p2 = jacobian.pinverse().t();
		},1,false);

		total1 += taken1;
		total2 += taken2;
	}

	cout << "matrix mul" << endl;
	cout << "total taken: " << total1 << "ms avg. " << (total1 / double (N)) << "ms"<< endl;

	cout << "pinverse()" << endl;
	cout << "total taken: " << total2 << "ms avg. " << (total2 / double (N)) << "ms"<< endl;
}
