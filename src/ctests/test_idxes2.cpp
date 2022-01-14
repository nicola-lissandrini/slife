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

#define PCL_SIZE 40000
#define BATCH_SIZE 20

int main ()
{
	Tensor pointcloud = torch::rand ({PCL_SIZE,3}, kFloat);
	makeNan (pointcloud, PCL_SIZE * 0.5);

	int left = BATCH_SIZE;
	int used = 0;
	Tensor idxes = torch::empty ({0}, kLong);
	Tensor perm = torch::randperm (pointcloud.size(0));
	while (left > 0) {
		Tensor selectPerm =  perm.slice (0, used, used + left);
		Tensor selectPointcloud = pointcloud.index ({selectPerm, Ellipsis});
		Tensor currValidIdxes = selectPerm.index ({selectPointcloud.isfinite ().sum(1) > 0});
		idxes = torch::cat ({idxes, currValidIdxes});
		used += left;
		left = BATCH_SIZE - idxes.size(0);
		COUTN(currValidIdxes);
		COUTN(idxes);
	}
}
