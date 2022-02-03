#include "slife/optimizer.h"
#include <torch/csrc/autograd/profiler.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace lietorch;

// Instantiate templates

// Position
template
class PointcloudMatch<Position>;
template
class Optimizer<Position, PointcloudMatch<Position>>;

// Quaternion R4
template
class PointcloudMatch<QuaternionR4>;
template
class Optimizer<QuaternionR4, PointcloudMatch<QuaternionR4>>;

// Quaternion
template
class PointcloudMatch<Quaternion>;
template
class Optimizer<Quaternion, PointcloudMatch<Quaternion>>;

// Pose3R4
template
class PointcloudMatch<Pose3R4>;
template
class Optimizer<Pose3R4, PointcloudMatch<Pose3R4>>;


template
class PointcloudMatch<Pose>;
template
class Optimizer<Pose, PointcloudMatch<Pose>>;

template<class LieGroup, class TargetCostFunction>
LieGroup Optimizer<LieGroup, TargetCostFunction>::getInitialValue()
{
	switch (params.initializationType) {
	case INITIALIZATION_IDENTITY:
		return initializations.identity;
	case INITIALIZATION_LAST:
		return initializations.lastResult;
	default:
		assert (false && "Initialization not supported");
	}

	return LieGroup ();
}

template<class LieGroup, class TargetCostFunction>
bool Optimizer<LieGroup,TargetCostFunction>::isReady() const {
	return costFunctionPtr->isReady () && !params.disable;
}

template<class LieGroup, class TargetCostFunction>
void Optimizer<LieGroup, TargetCostFunction>::reset () {
	initializations.lastResult = LieGroup::Identity ();
}

template<class LieGroup, class TargetCostFunction>
void Optimizer<LieGroup, TargetCostFunction>::optimize (LieGroup &estimate, History &history)
{
	LieGroup state = getInitialValue ();
	LieGroup nextState;
	bool terminationCondition = false;
	int iterations = 0;
	double taken, totalTaken = 0;

	history.clear ();

	COUTN(getInitialValue ());

	while (!terminationCondition) {
		if (params.recordHistory)
			history.push_back(state);

		PROFILE_N_EN(taken, [&]{
			auto gradient = costFunction()->gradient (state);

			/// OLD TRICK DISABLED nextState = state - gradient * (params.stepSizes / gradient.coeffs.norm().sqrt());
			nextState = state - gradient * params.stepSizes;
			terminationCondition = (nextState.dist(state, params.normWeights) < params.threshold).item().toBool() ||
							   (iterations >= params.maxIterations).item().toBool ();

			state = nextState;

			iterations++;

		}, 1, false);
		totalTaken += taken;
	}

	initializations.lastResult = state;
	estimate = state.coeffs.clone ();
	COUTN(state);
	cout << "total taken: " << totalTaken << "ms avg. " << (totalTaken / double (iterations)) << "ms"<< endl;
	cout << "Possible target Hz " << 1000/totalTaken << endl;
}

template<class LieGroup>
PointcloudMatch<LieGroup>::PointcloudMatch (const Landscape::Params::Ptr &landscapeParams,
								    const typename Params::Ptr &pointcloudMatchParams):
	CostFunction<LieGroup>(pointcloudMatchParams),
	landscape(*landscapeParams)
{
	flags.addFlag("old_pointcloud");
	flags.addFlag("new_pointcloud");

	sumOut = [] (const Tensor &t) { return t.sum(0); };
}

template<class LieGroup>
Pointcloud PointcloudMatch<LieGroup>::oldPointcloudBatch (bool clipUninformative) const
{
	if (!params().stochastic)
		return oldPcl;

	if (clipUninformative) {
		Tensor oldBatchValidIdxes = landscape.selectInformativeIndexes (landscape.getBatchIndexes (), oldPcl);
		return oldPcl.index ({oldBatchValidIdxes, Ellipsis});
	} else
		return oldPcl.index ({landscape.getBatchIndexes (), Ellipsis});
}

template<class LieGroup>
typename PointcloudMatch<LieGroup>::Tangent
PointcloudMatch<LieGroup>::gradient (const LieGroup &x)
{
	Pointcloud predicted;
	Tangent totalGradient;
	Tensor landscapeGradient, jacobian;
	//autograd::profiler::RecordProfile rp("/home/nicola/new.trace");

	landscape.shuffleBatchIndexes ();
	Tensor old = oldPointcloudBatch ();

	predicted = x * old;

	if (params().reshuffleBatchIndexes)
		landscape.shuffleBatchIndexes ();

	landscapeGradient = landscape.gradient(predicted);
	totalGradient = x.differentiate(landscapeGradient, predicted, sumOut, jacobian);

	return totalGradient;
}

template<class LieGroup>
typename PointcloudMatch<LieGroup>::Vector
PointcloudMatch<LieGroup>::value (const LieGroup &x)
{
	landscape.shuffleBatchIndexes ();
	Pointcloud predicted = x * oldPointcloudBatch ();
	Tensor totalValue = torch::zeros ({1}, kFloat);

	for (int i = 0; i < predicted.size(0); i++) {
		const Tensor &curr = predicted[i];
		totalValue += landscape.value (curr);
	}

	return totalValue;
}

template<class LieGroup>
bool PointcloudMatch<LieGroup>::isReady() const {
	return flags.isReady();
}

template<class LieGroup>
void PointcloudMatch<LieGroup>::updatePointcloud(const Pointcloud &pointcloud)
{
	if (flags["new_pointcloud"]) {
		flags.set("old_pointcloud");
		oldPcl = landscape.getPointcloud();
	}

	flags.set("new_pointcloud");
	landscape.setPointcloud (pointcloud);
}

template<class LieGroup>
Pointcloud PointcloudMatch<LieGroup>::getPointcloud () const {
	return landscape.getPointcloudBatch ();
}

template<>
TestFcn PointcloudMatch<Position>::getCostLambda (Test::Type type)
{
	switch (type) {
	case Test::TEST_COST_VALUES:
		return [this] (const Tensor &p) -> Tensor { return this->value(Position(p)); };
	case Test::TEST_COST_GRADIENT:
		return [this] (const Tensor &p) -> Tensor { return this->gradient(Position(p)).coeffs; };
	default:
		return TestFcn ();
	}
}

template<>
TestFcn PointcloudMatch<Pose3R4>::getCostLambda (Test::Type type)
{
	switch (type) {
	case Test::TEST_COST_VALUES:
		return [this] (const Tensor &p) -> Tensor { return this->value(Pose3R4(p,QuaternionR4())); };
	case Test::TEST_COST_GRADIENT:
		return [this] (const Tensor &p) -> Tensor { return this->gradient(Pose3R4(p,QuaternionR4())).coeffs.slice(0, 0, 3); };
	default:
		return TestFcn ();
	}
}

template<class LieGroup>
TestFcn PointcloudMatch<LieGroup>::getCostLambda (Test::Type type) {
	assert (false && "Cost function test not implemented for this LieGroup");
}


template<class LieGroup>
Tensor PointcloudMatch<LieGroup>::test (Test::Type type)
{
	if (!flags.isReady())
		return Tensor ();

	int testTensorDim;
	TestFcn testTensorFcn;

	switch (type) {
	case Test::TEST_LANDSCAPE_VALUES:
		testTensorDim = D_1D;
		testTensorFcn = [this] (const Tensor &p) -> Tensor { return this->landscape.value(p); };
		break;
	case Test::TEST_LANDSCAPE_GRADIENT:
		testTensorDim = D_3D;
		testTensorFcn = [this] (const Tensor &p) -> Tensor { return this->landscape.gradient(p); };
		break;
	case Test::TEST_COST_VALUES:
		testTensorDim = D_1D;
		testTensorFcn = getCostLambda (type);
		break;
	case Test::TEST_COST_GRADIENT:
		testTensorDim = D_3D;
		testTensorFcn = getCostLambda (type);
		break;
	case Test::TEST_NONE:
	default:
		return Tensor ();
	}

	Tensor testGrid = tester->getTestGrid();
	const int gridSize = tester->getTestGridSize();
	Tensor values;// = torch::empty ({testGrid.size(0), testTensorDim}, kFloat);

	if (type == Test::TEST_LANDSCAPE_GRADIENT ||
			type == Test::TEST_LANDSCAPE_VALUES)
		values = testTensorFcn(testGrid);
	else  {
		values = torch::empty ({testGrid.size(0), testTensorDim}, kFloat);
		for (int i = 0; i < testGrid.size(0); i++) {
			Tensor currentPoint = testGrid[i];
			Tensor value = testTensorFcn(currentPoint.unsqueeze(0));
			values[i] = value.squeeze();
		}
	}

	if (type == Test::TEST_LANDSCAPE_GRADIENT ||
			type == Test::TEST_COST_GRADIENT) {
		Tensor ret = values;//.reshape({gridSize * gridSize, testTensorDim});

		return ret;
	}
	else
		return values.reshape({gridSize, gridSize});
}

























