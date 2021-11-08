#include "lietorch/pose.h"
#include "../../sparcslib/include/profiling.h"

using namespace lietorch;
using namespace std;
using namespace torch;

#define N 40000

int main ()
{
	Pose pose(Position ({1,2,3}),
			Rotation ({1,1,0,0}));
	Pose pose2(Position ({1,1,1}),
			 Rotation (0, 0, -0.9589243, 0.2836622));
	Tensor pcl = torch::tensor ({{1.,2.,3.},{4.,5.,6.}}, kFloat);


	cout << pose2 * pcl << endl;
	return 0;
}
