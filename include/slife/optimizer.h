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
public:
	struct Params {
		virtual ~Params () = default;
		DEF_SHARED(Params)
	};

	using Tangent = typename LieGroup::Tangent;
	using Coeffs = typename LieGroup::DataType;
	using Vector = typename LieGroup::Vector;


	CostFunction (const typename Params::Ptr &params);

	virtual Vector value (const LieGroup &x) = 0;
	virtual Tangent gradient (const LieGroup &x) = 0;

	DEF_SHARED(CostFunction)

protected:
	typename Params::Ptr paramsData;
};

template<class LieGroup>
CostFunction<LieGroup>::CostFunction  (const typename Params::Ptr &params):
	paramsData(params)
{}

template<class LieGroup>
class Optimizer
{
	using Vector = typename LieGroup::Vector;
	using Coeffs = typename LieGroup::DataType;

public:
	struct Params {
		torch::Tensor stepSizes;
		torch::Tensor threshold;

		DEF_SHARED(Params)
	};

private:
	Params params;
	ReadyFlagsStr flags;
	typename CostFunction<LieGroup>::Ptr costFunction;

public:
	Optimizer (const typename Params::Ptr &_params,
			 const typename CostFunction<LieGroup>::Ptr &_costFunction):
		params(*_params),
		costFunction(_costFunction)
	{}

	LieGroup optimize (const LieGroup &initialValue);

	DEF_SHARED(Optimizer)
};

class PointcloudMatch : public CostFunction<lietorch::Pose>
{
public:
	struct Params : public CostFunction<lietorch::Pose>::Params {
		int batchSize;

		DEF_SHARED(Params)
	};

protected:
	Landscape landscape;
	ReadyFlagsStr flags;
	Pointcloud oldPcl;

	Params &params() const {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}

public:
	PointcloudMatch (const Landscape::Params::Ptr &landscapeParams,
				  const Params::Ptr &pointcloudMatchParams);

	Vector value (const lietorch::Pose &x);
	Tangent gradient (const lietorch::Pose &x);

	void updatePointcloud (const Pointcloud &pointcloud);

	Tensor test (Test::Type type);

	DEF_SHARED (PointcloudMatch)
};

using PoseOptimizer = Optimizer<lietorch::Pose>;

#endif // LOCALIZE_H
