#include "slife/optimizer.h"

using namespace std;
using namespace torch;

template<class LieGroup>
void Optimizer<LieGroup>::initialize(const LieGroup &initialValue) {
	flags.set("algorithm_initialized");
	state = initialValue;
}

template<class LieGroup>
LieGroup Optimizer<LieGroup>::optimize (const LieGroup &initialValue)
{
	bool terminationCondition = false;
	int iterations = 0;

	while (!terminationCondition) {
		LieGroup nextState = state - params.step_size * costFunction->gradient ();

		terminationCondition = ((nextState - state).norm () < params.threshold) ||
						   (iterations >= params.maxIterations);

		state = nextState;

		iterations++;
	}

	return state;
}


Optimizer::Optimizer(const Landscape::Params::Ptr &landscapeParams,
				 const Params::Ptr &optimizerParams):
	landscape(landscapeParams),
	paramsData(optimizerParams)
{
	flags.addFlag("algorithm_initialized");
	flags.addFlag("old_pointcloud");
	flags.addFlag("new_pointcloud");
}

void Optimizer::initialize(const Tensor &initialValue) {
	flags.set("algorithm_initialized");
	state = initialValue;
}

Tensor Optimizer::optimize ()
{
	if (!flags.isReady())
		return Tensor ();

	while (!terminationCondition ())
		iterationStep ();

	return Tensor ();
}

void Optimizer::updatePointcloud (const Tensor &pointcloud)
{
	if (flags["new_pointcloud"]) {
		flags.set("old_pointcloud");
		oldPointcloud = landscape.getPointcloud();
	}

	flags.set("new_pointcloud");
	landscape.setPointcloud(pointcloud);
}

Tensor Optimizer::test (Test::Type type)
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
	case Test::TEST_LANDSCAPE_GRADIENT:
		testTensorDim = D_3D;
		testTensorFcn = [this] (const Tensor &p) -> Tensor { return this->landscape.gradient(p); };
		break;
	default:
		return Tensor ();
	}

	Tensor testGrid = tester->getTestGrid();
	const int gridSize = tester->getTestGridSize();
	Tensor values = torch::empty ({testGrid.size(0), testTensorDim}, kFloat);

	for (int i = 0; i < testGrid.size(0); i++) {
		Tensor currentPoint = testGrid[i];

		Tensor value = testTensorFcn(currentPoint);
		values[i] = value.squeeze();
	}


	if (type == Test::TEST_LANDSCAPE_GRADIENT) {
		Tensor ret = values.reshape({gridSize * gridSize, testTensorDim});

		return ret;
	}
	else
		return values.reshape({gridSize, gridSize});
}
*/

