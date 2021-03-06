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
 * Approximate: int_Rn f(w) e^(-||w - x||^2/2radius) dw
 ***/
class Smoother
{
public:
	struct Params {
		int dim;
		int samplesCount;
		float radius;
	};

protected:
	Params params;

public:
	using Fcn = std::function<torch::Tensor(torch::Tensor)>;

	Smoother (int dim, int samplesCount, float radius);

	virtual torch::Tensor evaluate (const Fcn &f, const torch::Tensor &x) = 0;

	DEF_SHARED(Smoother)
};


class MontecarloSmoother : public Smoother
{
public:
	MontecarloSmoother (int dim, int samplesCount, float radius):
		Smoother (dim, samplesCount, radius)
	{}

	torch::Tensor evaluate (const Fcn &f, const torch::Tensor &x);

	DEF_SHARED(MontecarloSmoother)
};


// Compute a riemann summ approximation of the integral
class RiemannSmoother : public Smoother
{
public:
	RiemannSmoother (int dim, int samplesCount, float radius):
		Smoother (dim, samplesCount, radius)
	{}

	// !!!TODO
	torch::Tensor evaluate (const Fcn &f, const at::Tensor &x);
};

class Landscape
{
public:
	static constexpr int Dim = D_3D;

	struct Params {
		Scalar measureRadius;
		Scalar smoothRadius;
		int precision;

		DEF_SHARED(Params)
	};


private:
	Smoother::Fcn valueLambda, gradientLambda;

	Params params;
	Pointcloud pointcloud;
	ReadyFlagsStr flags;
	Smoother::Ptr smoother;
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

















