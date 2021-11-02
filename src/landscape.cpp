#include "slife/landscape.h"
#include <torch/csrc/autograd/profiler.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;

Landscape::Landscape (const Params::Ptr &_params):
	paramsData(_params),
	montecarlo(Dim, _params->precision, _params->smoothRadius)
{
	assert (params().measureRadius > params().smoothRadius && "Smooth radius must be smaller than measure radius");
	smoothGain = getSmoothGain ();
	flags.addFlag ("pointcloud_set", true);

	valueLambda = [this] (const torch::Tensor &p) -> torch::Tensor  {
		return preSmoothValue (p);
	};

	gradientLambda = [this] (const torch::Tensor &p) -> torch::Tensor  {
		return preSmoothGradient(p);
	};
}


void Landscape::setPointcloud(const Tensor &_pointcloud)
{
	pointcloud = _pointcloud.unsqueeze(1);
	flags.set("pointcloud_set");
}

Tensor Landscape::peak (const Tensor &v) const {
	return (- v * 0.5 / pow (params().measureRadius,2));
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
		return montecarlo.evaluate (valueLambda, p);
}

Tensor Landscape::preSmoothGradient (const Tensor &p) const
{
	Tensor pointcloudDiff = p - pointcloud;
	Tensor distToPointcloud = pointcloudDiff.pow(2).sum(2);
	Tensor collapsedDist, idxes;
	tie (collapsedDist, idxes) = distToPointcloud.min (0);

	Tensor collapsedDiff = pointcloudDiff.permute({1,0,2}).index({torch::arange(pointcloudDiff.size(1)),idxes,Ellipsis});

	return collapsedDiff / pow (params().measureRadius,2) * smoothGain;
}

Tensor Landscape::gradient (const Tensor &p)
{
	if (!flags.isReady())
		return Tensor ();

	if (pointcloud.size (0) == 0)
		return Tensor();
	else
		return montecarlo.evaluate(gradientLambda, p);
}

float Landscape::getNoAmplificationGain () const {
	return 0.5*M_SQRT2*pow(params().measureRadius, 2)/
			(pow(M_PI, 1.5)*pow(params().smoothRadius, 3)*(2*pow(params().measureRadius, 2) - 3*pow(params().smoothRadius, 2)));
}

float Landscape::getSmoothGain () const {
	return pow (2 * M_PI * pow (params().smoothRadius,2), 1.5) * getNoAmplificationGain();
}

GaussianMontecarlo::GaussianMontecarlo(int dim, int samplesCoount, float variance):
	params({dim, samplesCoount, variance})
{
	prealloc.xEval = torch::empty({params.samplesCount, params.dim}, kFloat);
	prealloc.xVar = torch::empty({params.samplesCount, params.dim}, kFloat);
}

Tensor GaussianMontecarlo::evaluate(const function<Tensor (Tensor)> &f, const Tensor &center)
{
	prealloc.xVar = torch::normal (0.0, params.variance, {params.samplesCount, params.dim});
	prealloc.xEval = prealloc.xVar + center.expand ({params.samplesCount, params.dim});

	return f(prealloc.xEval).mean (0);
}
















