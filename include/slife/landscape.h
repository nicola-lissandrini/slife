#ifndef LANDSCAPE_H
#define LANDSCAPE_H

#include <functional>
#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include "common.h"

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
		float measureRadius;
		float smoothRadius;
		int precision;

		DEF_SHARED (Params)
	};
private:
	GaussianMontecarlo::Fcn valueLambda, gradientLambda;

	Params::Ptr paramsData;
	torch::Tensor pointcloud;
	ReadyFlags<std::string> flags;
	GaussianMontecarlo montecarlo;
	float smoothGain;

	struct {
		torch::Tensor pointcloudDiff;
		torch::Tensor distToPoincloud;
		torch::Tensor collapsedDist;
		torch::Tensor collapsedDiff;
		torch::Tensor idxes;
	} prealloc;


	const Params &params () const {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}

	torch::Tensor peak (const torch::Tensor &v) const;
	torch::Tensor preSmoothValue (const torch::Tensor &p) const;
	torch::Tensor preSmoothGradient (const torch::Tensor &p) const;

	float getNoAmplificationGain () const;
	float getSmoothGain () const;


public:
	Landscape (const Params::Ptr &_params);

	void setPointcloud (const torch::Tensor &_pointcloud);

	torch::Tensor value (const torch::Tensor &p);
	torch::Tensor gradient (const torch::Tensor &p);

	DEF_SHARED(Landscape)
};


#endif // LANDSCAPE_H

















