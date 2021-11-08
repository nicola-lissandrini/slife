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
	Pose pose2(Position ({1,2,3}),
			Rotation ());
	Position3 t = pose.translation();
	cout << (pose * pose2).toString() << endl;


	cout << t.toString()<< endl;
	return 0;
}
