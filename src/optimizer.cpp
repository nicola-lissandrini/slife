#include "slife/optimizer.h"

using namespace std;
using namespace torch;


Optimizer::Optimizer(const Landscape::Params::Ptr &landscapeParams,
				 const Params::Ptr &optimizerParams):
	landscape(landscapeParams),
	paramsData(optimizerParams)
{
	// flags.addFlag("algorithm_initialized");
	flags.addFlag("first_pointcloud");
}

void Optimizer::updatePointcloud (const Tensor &pointcloud)
{
	flags.set("first_pointcloud");
	landscape.setPointcloud(pointcloud);
}

Tensor Optimizer::optimize ()
{
	return Tensor ();
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

	{
		//autograd::profiler::RecordProfile a("/home/nicola/values.trace");
		for (int i = 0; i < testGrid.size(0); i++) {
			Tensor currentPoint = testGrid[i];

			Tensor value = testTensorFcn(currentPoint);
			values[i] = value.squeeze();
		}
	}


	if (type == Test::TEST_LANDSCAPE_GRADIENT) {
		Tensor ret = values.reshape({gridSize * gridSize, testTensorDim});

		return ret;
	}
	else
		return values.reshape({gridSize, gridSize});
}


