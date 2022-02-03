#include "slife/landscape.h"
#include <torch/csrc/autograd/profiler.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;

Landscape::Landscape (const Params &_params):
	params(_params)
{
	assert (params.measureRadius > params.smoothRadius && "Smooth radius must be smaller than measure radius");
	smoothGain = getSmoothGain ();
	flags.addFlag ("pointcloud_set", true);
	smoother = make_shared<MontecarloSmoother> (+Dim, _params.precision, _params.smoothRadius);

	valueLambda = [this] (const torch::Tensor &p) -> torch::Tensor  {
		return preSmoothValue (p);
	};

	gradientLambda = [this] (const torch::Tensor &p) -> torch::Tensor  {
		return preSmoothGradient(p);
	};

	// Init indexing grid
	auto xyGrid = torch::meshgrid({torch::arange(0,params.batchSize),
							 torch::arange(0,params.precision)});
	xGrid = xyGrid[0].reshape({1,-1});
	yGrid = xyGrid[1].reshape({1,-1});
}

void Landscape::setPointcloud (const Pointcloud &_pointcloud)
{
	pointcloud = _pointcloud.slice (0, 0, nullopt, params.decimation).clone ();

	flags.set ("pointcloud_set");
}

Pointcloud Landscape::getPointcloud() const {
	return pointcloud;
}

void Landscape::shuffleBatchIndexes ()
{
	int left = params.batchSize;
	int used = 0;
	Tensor permutation = torch::randperm (pointcloud.size(0));
	batchIndexes = torch::empty ({0}, kLong);
	while (left > 0) {
		Tensor selectedPermutation = permutation.slice (0, used, used + left);
		Tensor currentValidIdxes = selectInformativeIndexes (selectedPermutation, pointcloud);
		batchIndexes = torch::cat ({batchIndexes, currentValidIdxes});

		used += left;
		left = params.batchSize - batchIndexes.size(0);
	}
}

Tensor Landscape::selectInformativeIndexes(const Tensor &indexes, const Pointcloud &pointcloud) const
{
	Tensor selectedPointcloud = pointcloud.index ({indexes, Ellipsis});
	return indexes.index ({(selectedPointcloud.isfinite ().sum(1) > 0)
					   .logical_and (selectedPointcloud.norm(2,1) > params.clipArea.min)
					   .logical_and (selectedPointcloud.norm(2,1) < params.clipArea.max)});
}

Tensor Landscape::getBatchIndexes() const {
	return batchIndexes;
}

Tensor Landscape::getPointcloudBatch() const
{
	if (!params.stochastic)
		return pointcloud;

	return pointcloud.index ({batchIndexes, Ellipsis});
}

Tensor Landscape::peak (const Tensor &v) const {
	return (- v * 0.5 / pow (params.measureRadius,2));
}

Tensor Landscape::preSmoothValue (const Tensor &p) const
{
	Tensor distToMeasures = (p - pointcloud).pow(2).sum(2);

	return peak (distToMeasures.index({distToMeasures.argmin(0)[0], Ellipsis})) * smoothGain;
}

Tensor Landscape::value(const Tensor &p)
{
	if (!flags.isReady())
		return Tensor ();

	if (pointcloud.size (0) == 0)
		return Tensor();
	else
		return smoother->evaluate (valueLambda, p);
}

Tensor Landscape::preSmoothGradient (const Tensor &p) const
{
	Tensor pointcloudCurrent = getPointcloudBatch ();
	Tensor pointcloudDiff = p - pointcloudCurrent.unsqueeze (1).unsqueeze(2);
	Tensor distToPointcloud = pointcloudDiff.pow(2).sum(3);
	Tensor collapsedDist, idxes;

	tie (collapsedDist, idxes) = distToPointcloud.min (0);

	Tensor collapsedDiff = pointcloudDiff.permute({1,2,0,3})
					   .index({xGrid.slice (1, 0, idxes.numel()),
							 yGrid.slice (1, 0, idxes.numel()),
							 idxes.reshape({1,-1}),
							 Ellipsis})
					   .reshape({-1, params.precision, 3});

	return collapsedDiff / pow (params.measureRadius,2) * smoothGain;
}

Tensor Landscape::gradient (const Tensor &p)
{
	if (!flags.isReady())
		return Tensor ();

	if (pointcloud.size (0) == 0)
		return Tensor();
	else
		return smoother->evaluate(gradientLambda, p);
}

float Landscape::getNoAmplificationGain () const {
	return 0.5*M_SQRT2*pow(params.measureRadius, 2)/
			(pow(M_PI, 1.5)*pow(params.smoothRadius, 3)*(2*pow(params.measureRadius, 2) - 3*pow(params.smoothRadius, 2)));
}

float Landscape::getSmoothGain () const {
	return pow (2 * M_PI * pow (params.smoothRadius,2), 1.5) * getNoAmplificationGain();
}

Smoother::Smoother (int dim, int samplesCount, float variance):
	params({dim, samplesCount, variance})
{
}

Tensor MontecarloSmoother::evaluate(const Fcn &f, const Tensor &x)
{
	Tensor xVar = torch::normal (0.0, params.radius, {params.samplesCount, params.dim});
	Tensor xEval = xVar + x.unsqueeze(1);

	return f(xEval).mean (1);
}
















