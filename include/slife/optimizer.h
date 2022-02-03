#ifndef LOCALIZE_H
#define LOCALIZE_H


#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>
#include <type_traits>
#include <functional>

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
	virtual bool isReady () const = 0;

	DEF_SHARED(CostFunction)

protected:
	ReadyFlagsStr flags;
	typename Params::Ptr paramsData;
};

template<class LieGroup>
CostFunction<LieGroup>::CostFunction  (const typename Params::Ptr &params):
	paramsData(params)
{}

#define INHERIT_TRAITS(_Base) \
	using Base = _Base; \
	using Tangent = typename Base::Tangent;\
	using Coeffs = typename Base::Coeffs;\
	using Vector = typename Base::Vector;\
	using Base::paramsData;


using TestFcn = std::function<Tensor(Tensor)>;

template<class LieGroup>
class PointcloudMatch : public CostFunction<LieGroup>
{
	INHERIT_TRAITS(CostFunction<LieGroup>)

public:
	struct Params : public CostFunction<LieGroup>::Params {
		int batchSize;
		bool stochastic;
		bool reshuffleBatchIndexes;
		DEF_SHARED(Params)
	};

protected:
	Landscape landscape;
	ReadyFlagsStr flags;
	Pointcloud oldPcl;

	lietorch::OpFcn sumOut;


	Params &params () {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}
	const Params &params () const {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}

public:
	PointcloudMatch (const Landscape::Params::Ptr &landscapeParams,
				  const typename Params::Ptr &pointcloudMatchParams);

	Vector value (const LieGroup &x);
	Tangent gradient (const LieGroup &x);

	void updatePointcloud (const Pointcloud &pointcloud);
	Pointcloud getPointcloud () const;
	bool isReady () const;

	// Test
	Tensor test (Test::Type type);
	Landscape::Params::Ptr getLandscapeParams () const;
	Pointcloud oldPointcloudBatch (bool clipUninformative = true) const;

	// Cost testing only implemented for Position or Pose
	TestFcn getCostLambda (Test::Type);

	DEF_SHARED (PointcloudMatch)
};


template<class LieGroup, class TargetCostFunction>
class Optimizer
{
	using Vector = typename LieGroup::Vector;
	using Coeffs = typename LieGroup::DataType;

public:
	using History = std::vector<LieGroup>;

	enum InitializationType {
		INITIALIZATION_IDENTITY = 0,
		INITIALIZATION_LAST
	};

	struct Params {
		torch::Tensor stepSizes;
		torch::Tensor normWeights;
		torch::Tensor threshold;
		InitializationType initializationType;
		torch::Tensor maxIterations;
		bool recordHistory;
		bool disable;

		DEF_SHARED(Params)
	};

private:
	Params params;
	typename TargetCostFunction::Ptr costFunctionPtr;

	struct {
		LieGroup identity;
		LieGroup lastResult;
	} initializations;

	LieGroup getInitialValue ();

public:
	Optimizer (const typename Params::Ptr &_params,
			 const typename TargetCostFunction::Ptr &_costFunctionPtr):
		params(*_params),
		costFunctionPtr(_costFunctionPtr)
	{}

	void optimize (LieGroup &estimate, History &history);
	bool isReady () const;
	void reset();
	void disable () {
		params.disable = true;
	}
	void enable () {
		params.disable = false;
	}
	typename TargetCostFunction::Ptr costFunction () {
		return costFunctionPtr;
	}

	DEF_SHARED(Optimizer)
};


// Supported optimziation types
template<class LieGroup>
using PointcloudMatchOptimizer = Optimizer<LieGroup, PointcloudMatch<LieGroup>>;

using PositionOptimizer = PointcloudMatchOptimizer<lietorch::Position>;
using PoseOptimizer = PointcloudMatchOptimizer<lietorch::Pose3R4>;
#endif // LOCALIZE_H
