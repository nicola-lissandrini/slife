#include "lietorch/quaternion.h"
#include "../../sparcslib/include/profiling.h"

using namespace lietorch;
using namespace std;
using namespace torch;

#define N 40000

int main ()
{
	UnitQuaternionR4 a(0.0868241, 0.0868241, 0.0075961, 0.9924039);
	Tensor v = torch::tensor({{1., 1., 1.}, {1.,2.,3.}, {3.,2.,1.}, {1., 1., 1.}}, kFloat);
	cpou
	cout << a * v << endl;
	return 0;
}
