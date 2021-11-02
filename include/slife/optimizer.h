#ifndef LOCALIZE_H
#define LOCALIZE_H


#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include "landscape.h"
#include "test.h"

class Optimizer
{
public:
	struct Params {
		float translationRate;
		float rotationRate;

		DEF_SHARED (Params)
	};

private:
	Params::Ptr paramsData;
	Landscape landscape;
	ReadyFlags<std::string> flags;

	const Params &params () const {
		return *paramsData;
	}

public:
	Optimizer (const Landscape::Params::Ptr &landscapeParams,
			 const Params::Ptr &optimizerParams);

	void initialize (const torch::Tensor &initialValue);
	virtual torch::Tensor optimize ();
	void updatePointcloud (const torch::Tensor &pointcloud);

	torch::Tensor test(Test::Type type);

	DEF_SHARED(Optimizer)
};

#endif // LOCALIZE_H
