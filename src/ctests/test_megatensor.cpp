#include <torch/all.h>
#include "../../sparcsnode/include/sparcsnode/utils.h"

using namespace torch;
using namespace torch::indexing;
using namespace std;

int main ()
{
	Tensor pcl = torch::tensor ({{2, 6, 0},{6,5,0},{3,1,0}}, kFloat);
	COUTN(pcl);
	Tensor pEval = torch::tensor ({{5,4,0},{2,2,0},{1,7,0},{2,4, 0}}, kFloat);
	Tensor pVar = 0*torch::tensor({{0.1,0.1, 0.}, {-0.1, -0.1, 0.}, {0.1,-0.1, 0.},{-0.1,0.1, 0.}}, kFloat);
	Tensor pMonte = pVar + pEval.unsqueeze(1);
	Tensor pointcloudDiff = pMonte - pcl.unsqueeze(1).unsqueeze(2);
	Tensor distToPointcloud = pointcloudDiff.pow(2).sum(3).sqrt();
	COUTN(pointcloudDiff);
	COUTN(distToPointcloud);
	Tensor collapsedDist, idxes;
	tie(collapsedDist, idxes) = distToPointcloud.min (0);
	COUTN(collapsedDist);
	COUTN(idxes);
	Tensor pointsRange = torch::arange(0, pEval.size(0));
	Tensor monteRange = torch::arange(0,4);

	auto xy = torch::meshgrid({pointsRange, monteRange});
	COUTN(pMonte);
	COUTN(xy[0]);
	COUTN(xy[0].reshape({1, -1}));
	COUTN(xy[1].reshape({1, -1}));
	COUTN(idxes.reshape({1,-1}));
	COUTN(pointcloudDiff.permute({1,2,0,3}).index({xy[0].reshape({1,-1}),
										  xy[1].reshape({1,-1}),
										  idxes.reshape({1,-1}),
										  Ellipsis}).reshape({pEval.size(0),4,-1}));


/*
	Tensor pointcloudDiff = pEval - pcl.unsqueeze(1);
	Tensor distToPointcloud = pointcloudDiff.pow(2).sum(2);

	COUTN(pointcloudDiff);
	COUTN(distToPointcloud);
	Tensor collapsedDist, idxes;
	tie(collapsedDist, idxes) = distToPointcloud.min (0);
	COUTN(collapsedDist);
	COUTN(idxes);
	Tensor diffPermute = pointcloudDiff.permute({1,0,2});
	COUTN(diffPermute);
	Tensor collapsedDiff = diffPermute.index({torch::arange(pointcloudDiff.size(2)), idxes, Ellipsis});
	COUTN(collapsedDiff);*/
}
