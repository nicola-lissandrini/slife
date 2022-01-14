#include <torch/all.h>
#include "../../sparcsnode/include/sparcsnode/utils.h"
#include "../../sparcsnode/include/sparcsnode/profiling.h"


using namespace torch;
using namespace torch::indexing;

void makeNan (Tensor &pointcloud, int nanCount)
{
	Tensor nanPerm = torch::randperm (pointcloud.size(0)).slice (0, 0, nanCount);
	pointcloud.index_put_ ({nanPerm, Ellipsis}, Scalar(NAN));
}

#define BATCH_SIZE 20
#define QUANTI 1000
int main ()
{
	QUA;
	Tensor pointcloud = torch::rand ({40000,3}, kFloat);

	QUA;

	makeNan (pointcloud, 40000/100*50);

	Tensor validPointcloud = torch::empty ({0, pointcloud.size(1)});
	int left = BATCH_SIZE;
	int used = 0;
	int stepsRequired = 0;
	float taken;
	PROFILE_N(taken, [&]{
		autograd::profiler::RecordProfile aa("/home/nicola/idxes.trace");
		for (int i = 0; i < QUANTI; i++) {
			left = BATCH_SIZE;
			used = 0;
			stepsRequired = 0;
			Tensor totalPerm = torch::randperm (pointcloud.size(0));
			while (left > 0) {
				Tensor selectPerm = totalPerm.slice (0, used, used + left);
				Tensor selectPointcloud = pointcloud.index ({selectPerm, Ellipsis});
				Tensor validIdxes = selectPointcloud.isfinite ().sum(1).nonzero ();
				Tensor currValidPointcloud = selectPointcloud.index ({validIdxes}).view ({validIdxes.size(0), selectPointcloud.size(1)});
				validPointcloud = torch::cat ({validPointcloud, currValidPointcloud});
				used += left;
				left = BATCH_SIZE - validPointcloud.size(0);
				stepsRequired++;
			}
		}
	}, QUANTI);
	COUTN(stepsRequired);
	COUT("ogni volta un perm nuovo");
	PROFILE_N(taken, [&]{
		autograd::profiler::RecordProfile aa("/home/nicola/idxes2.trace");
		for (int i = 0; i < QUANTI; i++) {
			left = BATCH_SIZE;
			used = 0;
			stepsRequired = 0;
			while (left > 0) {
				Tensor selectPerm =  torch::randperm (pointcloud.size(0)).slice (0, 0, left);
				Tensor selectPointcloud = pointcloud.index ({selectPerm, Ellipsis});
				Tensor validIdxes = selectPointcloud.isfinite ().sum(1).nonzero ();
				Tensor currValidPointcloud = selectPointcloud.index ({validIdxes}).view ({validIdxes.size(0), selectPointcloud.size(1)});
				validPointcloud = torch::cat ({validPointcloud, currValidPointcloud});
				left = BATCH_SIZE - validPointcloud.size(0);
				stepsRequired++;
				used += currValidPointcloud.size(0);
			}
		}
	}, QUANTI);
	COUTN(stepsRequired);
}
