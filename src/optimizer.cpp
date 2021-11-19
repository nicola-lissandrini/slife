#include "slife/optimizer.h"

using namespace std;
using namespace torch;
using namespace lietorch;


template<class LieGroup>
LieGroup Optimizer<LieGroup>::optimize (const LieGroup &initialValue)
{
	LieGroup state = initialValue;
	bool terminationCondition = false;
	int iterations = 0;

	while (!terminationCondition) {
		LieGroup nextState = state - costFunction->gradient (state) * params.stepSizes;

		terminationCondition = ((nextState - state).norm () < params.threshold) ||
						   (iterations >= params.maxIterations);

		state = nextState;

		iterations++;
	}

	return state;
}

PointcloudMatch::PointcloudMatch (const Landscape::Params::Ptr &landscapeParams,
						    const Params::Ptr &pointcloudMatchParams):
	CostFunction(pointcloudMatchParams),
	landscape(*landscapeParams)
{
	flags.addFlag("old_pointcloud");
	flags.addFlag("new_pointcloud");
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
	cout << totalGradient.coeffs << endl;
	return totalGradient;
}

PointcloudMatch::Vector PointcloudMatch::value (const Pose &x)
{
	Pointcloud predicted = x * oldPcl.squeeze();
	Tensor totalValue = torch::zeros ({1}, kFloat);

	for (int i = 0; i < predicted.size(0); i++){
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
		testTensorFcn = [this] (const Tensor &p) -> Tensor { return this->gradient(Pose(p,Rotation())).coeffs; };
		break;
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


























