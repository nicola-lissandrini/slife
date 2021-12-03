#include "lietorch/rn.h"
#include <iostream>

using namespace std;
using namespace lietorch;
using namespace torch;

int main () {
	Rn<3> a({0,0,0});
	Tensor data = torch::tensor ({{1,2,3},{4,5,6}}, kFloat);

	cout << "a" << endl << a << endl;
	auto bubu = a * data;

	cout << (bubu - data).norm ()  << endl;

}
