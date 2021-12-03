#ifndef DEBUGGER_H
#define DEBUGGER_H

#include "sparcsnode/common.h"
#include <torch/torch.h>

class SlifeNode;

class Test
{
public:
	enum Type {
		TEST_NONE = 0,
		TEST_LANDSCAPE_VALUES,
		TEST_LANDSCAPE_GRADIENT,
		TEST_COST_VALUES,
		TEST_COST_GRADIENT
	};

private:

	std::shared_ptr<SlifeNode> nodePtr;

	struct Params {
		Range testGridRanges;
		std::string rangeTopic;
		Type testType;
		float zTestValue;
	} params;

	struct {
		torch::Tensor points;
		int xySize;
	} testGrid;

	void initTestGrid ();
	void initParams (XmlRpc::XmlRpcValue &xmlParams);

public:

	Test (XmlRpc::XmlRpcValue &xmlParams,
			const std::shared_ptr<SlifeNode> &_nodePtr);

	void publishRangeTensor (Type type, const torch::Tensor &tensor);
	torch::Tensor getTestGrid () const;
	int getTestGridSize () const;
	Type getType () const;

	DEF_SHARED(Test)
};

extern Test::Ptr tester;

#endif // DEBUGGER_H

