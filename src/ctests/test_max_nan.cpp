#include <torch/all.h>
#include "../../sparcsnode/include/sparcsnode/utils.h"

using namespace torch;
using namespace torch::indexing;
using namespace std;

#define BATCH_SIZE 10
#define PRECISION 2

auto xyGrid = torch::meshgrid({torch::arange(0,BATCH_SIZE),
						 torch::arange(0,PRECISION)});
auto xGrid = xyGrid[0].reshape({1,-1});
auto yGrid = xyGrid[1].reshape({1,-1});

Tensor mmm = torch::tensor({0.5}, kFloat);

Tensor preSmoothGradient (const Tensor &p, const Tensor &pointcloud)
{

	Tensor pointcloudDiff = p - pointcloud.unsqueeze(0).unsqueeze(2);
	Tensor distToPointcloud = pointcloudDiff.pow(2).sum(3);
	COUTN(distToPointcloud);
	Tensor collapsedDist, idxes;

	tie (collapsedDist, idxes) = distToPointcloud.min (0);
	auto boh = distToPointcloud < mmm;
	COUTN(boh);

	Tensor collapsedDiff = pointcloudDiff.permute({1,2,0,3})
					   .index({xGrid,
							 yGrid,
							 idxes.reshape({1,-1}),
							 Ellipsis})
					   .reshape({BATCH_SIZE,
							   PRECISION, -1});
	COUTN(collapsedDiff);
	return collapsedDiff;
}

Tensor gradient (const Tensor &x, const Tensor &pcl)
{
	Tensor xVar = torch::normal (0.0, 0.0, {PRECISION, 3});
	Tensor xEval = xVar + x.unsqueeze(1);

	return preSmoothGradient(xEval, pcl).mean (1);
}

int main ()
{
	vector<Tensor> pcl;


	torch::load (pcl, "/home/nicola/real_pcl");
	COUTN (pcl[0]);

/*	Tensor pcl = torch::rand({10,3}, kFloat);
	Tensor pEval = torch::tensor({{0,0,0}}, kFloat);

	pcl[3] *= NAN;

	COUTN(pEval);
	gradient(pEval, pcl);*/
}
