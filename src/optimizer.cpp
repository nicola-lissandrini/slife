#include "slife/optimizer.h"

using namespace std;
using namespace torch;
using namespace lietorch;


template<class LieGroup>
LieGroup Optimizer<LieGroup>::getInitialValue()
{
	switch (params.initializationType) {
	case INITIALIZATION_IDENTITY:
		return LieGroup ();
	default:
		assert (false && "Initialization not supported");
	}
}

template<class LieGroup>
LieGroup Optimizer<LieGroup>::optimize ()
{
	LieGroup state = getInitialValue ();
	bool terminationCondition = false;
	int iterations = 0;

	while (!terminationCondition) {
		if (params.recordHistory)
			history.push_back(state);

		typename LieGroup::Tangent gradient = costFunction->gradient (state);
		LieGroup nextState = state - gradient * params.stepSizes;

		COUTN(state);
		COUTN(gradient);
		COUTN(gradient*params.stepSizes);

		terminationCondition = ((nextState - state).norm () < params.threshold).item().toBool() ||
						   (iterations >= params.maxIterations).item().toBool ();

		state = nextState;

		iterations++;
	}

	return state;
}

template<class LieGroup>
vector<LieGroup> Optimizer<LieGroup>::getHistory() const {
	return history;
}

PointcloudMatch::PointcloudMatch (const Landscape::Params::Ptr &landscapeParams,
						    const Params::Ptr &pointcloudMatchParams):
	CostFunction(pointcloudMatchParams),
	landscape(*landscapeParams)
{
	flags.addFlag("old_pointcloud");
	flags.addFlag("new_pointcloud");
}

bool PointcloudMatch::isReady() const {
	return flags.isReady();
}

void PointcloudMatch::updatePointcloud(const Pointcloud &pointcloud)
{
	if (flags["new_pointcloud"]) {
		flags.set("old_pointcloud");
		oldPcl = landscape.getPointcloud();
	}
	flags.set("new_pointcloud");
	landscape.setPointcloud (pointcloud);
}

PointcloudMatch::Tangent PointcloudMatch::gradient(const Pose &x)
{
	Pointcloud predicted = x * oldPcl.squeeze();
	Tangent totalGradient;


	for (int i = 0; i < predicted.size(0); i++) {
		const Tensor &curr = predicted[i];

		totalGradient += x.differentiate(landscape.gradient (curr), curr);
	}

	return totalGradient;
}

PointcloudMatch::Vector PointcloudMatch::value (const Pose &x)
{
	Pointcloud predicted = x * oldPcl.squeeze();
	Tensor totalValue = torch::zeros ({1}, kFloat);

	for (int i = 0; i < predicted.size(0); i++) {
		const Tensor &curr = predicted[i];
		totalValue += landscape.value (curr);
	}

	return totalValue;
}

Tensor PointcloudMatch::test (Test::Type type)
{
	if (!flags.isReady())
		return Tensor ();

	int testTensorDim;
	function<Tensor(Tensor)> testTensorFcn;

	switch (type) {
	case Test::TEST_LANDSCAPE_VALUES:
		testTensorDim = D_1D;
		testTensorFcn = [this] (const Tensor &p) -> Tensor { return this->landscape.value(p); };
		break;
	case Test::TEST_COST_VALUES:
		testTensorDim = D_1D;
		testTensorFcn = [this] (const Tensor &p) -> Tensor { return this->value(Pose(p,Rotation())); };
		break;
	case Test::TEST_LANDSCAPE_GRADIENT:
		testTensorDim = D_3D;
		testTensorFcn = [this] (const Tensor &p) -> Tensor { return this->landscape.gradient(p); };
		break;
	case Test::TEST_COST_GRADIENT:
		testTensorDim = D_3D;
		testTensorFcn = [this] (const Tensor &p) -> Tensor { return this->gradient(Pose(p,Rotation())).coeffs.slice(0, 0, 3); };
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


























