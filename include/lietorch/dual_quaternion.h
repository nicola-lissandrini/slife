#ifndef DUAL_QUATERNION_H
#define DUAL_QUATERNION_H

#include "rn.h"
#include "quaternion.h"

namespace lietorch {

class DualQuaternion;
class DualTwist;

struct traits<DualQuaternion>
{
	static constexpr int Dim = 2 * LIETORCH_QUATERNION_DIM;
	static constexpr int ActDim = LIETORCH_POSITION_DIM;

	using Tangent = DualTwist;
	using Vector = torch::Tensor;
	using DataType = torch::Tensor;
};

struct traits<DualTwist>
{
	static constexpr int Dim = 2 * LIETORCH_QUATERNION_DIM;
};

}

#endif // DUAL_QUATERNION_H
