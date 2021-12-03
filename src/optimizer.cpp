#include "slife/optimizer.h"
#include <torch/csrc/autograd/profiler.h>

using namespace std;
using namespace torch;
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

// Pose
template
class PointcloudMatch<Pose>;
template
class Optimizer<Pose, PointcloudMatch<Pose>>;

// Todo: instantiate Quaternion, DualQuaternion

template<class LieGroup, class TargetCostFunction>
LieGroup Optimizer<LieGroup, TargetCostFunction>::getInitialValue()
{
	switch (params.initializationType) {
	case INITIALIZATION_IDENTITY:
		return LieGroup ();
	default:
		assert (false && "Initialization not supported");
	}

	return LieGroup ();
}

template<class LieGroup, class TargetCostFunction>
vector<LieGroup> Optimizer<LieGroup, TargetCostFunction>::getHistory() const {
	return history;
}

template<class LieGroup, class TargetCostFunction>
LieGroup Optimizer<LieGroup, TargetCostFunction>::optimize ()
{
	LieGroup state = getInitialValue ();
	LieGroup nextState;
	bool terminationCondition = false;
	int iterations = 0;
	double taken, totalTaken = 0;

	while (!terminationCondition) {
		if (params.recordHistory)
			history.push_back(state);
		//PROFILE(taken, [&]{
		nextState = state - costFunction()->gradient (state) * params.stepSizes;
		//});

		terminationCondition = (nextState.dist(state, params.normWeights) < params.threshold).item().toBool() ||
						   (iterations >= params.maxIterations).item().toBool ();

		state = nextState;

		iterations++;

		totalTaken += taken;
	}

	cout << "total taken: " << totalTaken << "ms avg. " << (totalTaken / double (iterations)) << "ms"<< endl;
	cout << "Possible target Hz " << 1000/totalTaken << endl;

	return state;
}

template<class LieGroup>
PointcloudMatch<LieGroup>::PointcloudMatch (const Landscape::Params::Ptr &landscapeParams,
						    const typename Params::Ptr &pointcloudMatchParams):
	CostFunction<LieGroup>(pointcloudMatchParams),
	landscape(*landscapeParams)
{
	flags.addFlag("old_pointcloud");
	flags.addFlag("new_pointcloud");
}

template<class LieGroup>
typename PointcloudMatch<LieGroup>::Tangent
PointcloudMatch<LieGroup>::gradient (const LieGroup &x)
{
	Pointcloud predicted;
	{
		autograd::profiler::RecordProfile rp("/home/nicola/predict.trace");
		predicted = x * oldPcl.squeeze();
	}
	Tangent totalGradient;
	{
		autograd::profiler::RecordProfile rp("/home/nicola/one.trace");
	for (int i = 0; i < predicted.size(0); i++) {
		//const Tensor &curr = ;
		//const Tensor &currGrad = ;

		totalGradient += x.differentiate(landscape.gradient(predicted[i]), predicted[i]);
	}
	}

	return totalGradient;
}

template<class LieGroup>
typename PointcloudMatch<LieGroup>::Vector
PointcloudMatch<LieGroup>::value (const LieGroup &x)
{
	Pointcloud predicted = x * oldPcl.squeeze();
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

template<>
TestFcn PointcloudMatch<Position>::getCostLambda (Test::Type type)
{
	switch (type) {
	case Test::TEST_COST_VALUES:
		return [this] (const Tensor &p) -> Tensor { return this->value(Position(p)); };
	case Test::TEST_COST_GRADIENT:
		return [this] (const Tensor &p) -> Tensor { return this->gradient(Position(p)).coeffs.slice(0, 0, 3); };
	default:
		return TestFcn ();
	}
}

template<>
TestFcn PointcloudMatch<Pose>::getCostLambda (Test::Type type)
{
	switch (type) {
	case Test::TEST_COST_VALUES:
		return [this] (const Tensor &p) -> Tensor { return this->value(Pose(p,QuaternionR4())); };
	case Test::TEST_COST_GRADIENT:
		return [this] (const Tensor &p) -> Tensor { return this->gradient(Pose(p,QuaternionR4())).coeffs.slice(0, 0, 3); };
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
	Tensor values = torch::empty ({testGrid.size(0), testTensorDim}, kFloat);

	float taken;
	PROFILE_N(taken,[&]{

	for (int i = 0; i < testGrid.size(0); i++) {
		Tensor currentPoint = testGrid[i];
		Tensor value = testTensorFcn(currentPoint);

		values[i] = value.squeeze();
	}

	}, testGrid.size(0));


	if (type == Test::TEST_LANDSCAPE_GRADIENT ||
			type == Test::TEST_COST_GRADIENT) {
		Tensor ret = values.reshape({gridSize * gridSize, testTensorDim});

		return ret;
	}
	else
		return values.reshape({gridSize, gridSize});
}



























