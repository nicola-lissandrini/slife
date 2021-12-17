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
	Tensor decimated, selected;
	Tensor validIdxes;

	decimated = _pointcloud.slice (0, 0, nullopt, params.decimation);

	validIdxes = torch::isfinite (decimated).sum (1)/*.logical_and (decimated.norm(2,1) < params.maximumDistance)*/.nonzero ();
	selected = decimated.index ({validIdxes}).view ({validIdxes.size(0), D_3D});

	pointcloud = selected;
	flags.set ("pointcloud_set");
}

/*
void Landscape::setPointcloud(const Pointcloud &_pointcloud)
{
	Tensor decimated, pointcloudTensor;

	double taken;
	ROS_WARN ("Decimate");
	PROFILE (taken,[&]{
		decimated = _pointcloud.slice (0, 0, nullopt, params.decimation);
	});
	ROS_WARN ("Finding only finite points both");
	PROFILE (taken,[&]{

		torch::Tensor validIdxes = (torch::isfinite(decimated)).sum(1).logical_and (decimated.norm (2,1) < params.maximumDistance).nonzero();

		pointcloudTensor = _pointcloud.index ({validIdxes}).view ({validIdxes.size(0), D_3D});
	});
	COUTN(pointcloudTensor.size(0))

	pointcloud = pointcloudTensor.unsqueeze(1);
	flags.set("pointcloud_set");
}
*/

Pointcloud Landscape::getPointcloud() const {
	return pointcloud;
}

void Landscape::shuffleBatchIndexes () {
	batchIndexes = torch::randperm (pointcloud.size(0)).slice (0,0,params.batchSize);
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
















