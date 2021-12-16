#ifndef DUAL_QUATERNION_H
#define DUAL_QUATERNION_H

#include "rn.h"
#include "quaternion.h"

namespace lietorch {

class DualQuaternion;
class DualTwist;

using Position = Position3;

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
	static constexpr int Dim = LIETORCH_POSITION_DIM + LIETORCH_ANGULAR_VELOCITY_DIM;

	using LieAlg = torch::Tensor;
	using LieGroup = DualQuaternion;
};

class DualTwist : public Tangent<DualTwist>
{
	using Base = Tangent<DualTwist>;
	
	using LinearVelocity = Position::Tangent;
	using AngularVelocity = Quaternion::Tangent;
	
public:
	LIETORCH_INHERIT_TANGENT_TRAITS
	
	using Base::coeffs;
	
	DualTwist (const LinearVelocity &_linear = LinearVelocity(), const AngularVelocity &_angular = AngularVelocity ()); 
	
	LieAlg generator (int i) const;
	LieAlg hat () const;
	DualQuaternion exp () const;
	DualTwist scale (const DataType &other) const;
	
	LinearVelocity linear () const;
	AngularVelocity angular () const;
	
};

class DualQuaternion : public LieGroup<DualQuaternion>
{
	using Base = LieGroup<DualQuaternion>;

	DataType realPart () const;
	DataType &realPart ();
	DataType dualPart () const;
	DataType &dualPart ();
	
public:
	LIETORCH_INHERIT_CONSTRUCTOR(DualQuaternion)
	LIETORCH_INHERIT_GROUP_TRAITS

	using Base::coeffs;

	DualQuaternion (const Position &_translation = Position(), const Quaternion &_rotation = Quaternion());
	DualQuaternion (const DataType &_realPart, const DataType &_dualPart);
	
	DualQuaternion inverse () const;
	DualTwist log () const;
	DualQuaternion compose (const DualQuaternion &other) const;
	DataType dist (const DualQuaternion &other, const DataType &weights) const;
	Vector act (const Vector &v) const;
	DualTwist differentiate (const Vector &outerGradient,
						const Vector &v,
						const OpFcn &op,
						const boost::optional<at::Tensor &> &jacobian) const;

	Position translation () const;
	void setTranslation (const Position &_position);
	Quaternion rotation () const;
	void setRotation (const Quaternion &_rotation);

};

#endif // DUAL_QUATERNION_H
