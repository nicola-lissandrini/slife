#ifndef LANDSCAPE_H
#define LANDSCAPE_H

#include <functional>
#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include "common.h"

using Pointcloud = torch::Tensor;
using Scalar = float_t;
using Tensor = torch::Tensor;

/***
 * Approximate: int_Rn f(w) e^(-||w - x||^2/2var) dw
 ***/
class GaussianMontecarlo
{
public:
	struct Params {
		int dim;
		int samplesCount;
		float variance;
	};

private:
	Params params;

	struct {
		torch::Tensor xVar;
		torch::Tensor xEval;
	} prealloc;

public:
	using Fcn = std::function<torch::Tensor(torch::Tensor)>;

	GaussianMontecarlo (int dim, int samplesCoount, float variance);

	torch::Tensor evaluate (const Fcn &f, const torch::Tensor &center);
};

class Landscape
{
public:
	static constexpr int Dim = D_3D;

	struct Params {
		Scalar measureRadius;
		Scalar smoothRadius;
		int precision;
	};

private:
	GaussianMontecarlo::Fcn valueLambda, gradientLambda;

	Params params;
	Pointcloud pointcloud;
	ReadyFlagsStr flags;
	GaussianMontecarlo montecarlo;
	float smoothGain;

	Tensor peak (const Tensor &v) const;
	Tensor preSmoothValue (const Tensor &p) const;
	Tensor preSmoothGradient (const Tensor &p) const;

	float getNoAmplificationGain () const;
	float getSmoothGain () const;


public:
	Landscape (const Params &_params);

	void setPointcloud (const Pointcloud &_pointcloud);
	Pointcloud getPointcloud () const;

	Tensor value (const Tensor &p);
	Tensor gradient (const Tensor &p);

	DEF_SHARED(Landscape)
};


#endif // LANDSCAPE_H

















