#ifndef LOCALIZE_H
#define LOCALIZE_H


#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include "landscape.h"
#include "lietorch/pose.h"
#include "test.h"

template<class LieGroup>
class CostFunction
{
	using Tangent = typename LieGroup::Tangent;
	using Scalar = typename LieGroup::Scalar;
	using Vector = typename LieGroup::Vector;

public:
	virtual Scalar value (const LieGroup &x) = 0;
	virtual Tangent gradient (const LieGroup &x) = 0;
};

template<class LieGroup>
using CostFunctionPtr = std::shared_ptr<CostFunction<LieGroup>>;

template<class LieGroup>
class Optimizer
{
	using CostFunctionSpecPtr = CostFunctionPtr<LieGroup>;
	using Vector = typename LieGroup::Vector;
	using Scalar = typename LieGroup::Scalar;

public:
	struct Params {
		Vector stepSizes;
		Scalar threshold;
	};

private:
	Params params;
	ReadyFlagsStr flags;
	CostFunctionSpecPtr costFunction;

public:
	Optimizer (const Params &_params,
			 const CostFunctionSpecPtr &_costFunction):
		params(_params),
		costFunction(_costFunction)
	{}

	LieGroup optimize (const LieGroup &initialValue);
};

class PointcloudMatch : public CostFunction<lietorch::Pose>
{
public:
	struct Params {

	};

protected:
	Landscape landscape;
	ReadyFlagsStr flags;
	Pointcloud oldPointcloud;

public:
	PointcloudMatch (const Landscape::Params &landscapeParams,
				  const Params &optimizerParams);

	Scalar value (const lietorch::Pose &x);
	Tangent gradient (const lietorch::Pose &x);

	void updatePointcloud (const Pointcloud &pointcloud);

	Tensor test (Test::Type type);
};

#endif // LOCALIZE_H
