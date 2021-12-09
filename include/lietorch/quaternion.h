#ifndef QUATERNION_H
#define QUATERNION_H

#include "rn.h"

namespace lietorch {

#define LIETORCH_QUATERNION_DIM 4
#define LIETORCH_POSITION_DIM 3
#define LIETORCH_ANGULAR_VELOCITY_DIM 3

class QuaternionR4;
class QuaternionR4Velocity;

class Quaternion;
class AngularVelocity;

namespace internal {

template<>
struct traits<QuaternionR4>
{
	static constexpr int Dim = LIETORCH_QUATERNION_DIM;
	static constexpr int ActDim = LIETORCH_POSITION_DIM;

	using Tangent = QuaternionR4Velocity;
	using Vector = torch::Tensor;
	using DataType = torch::Tensor;
};


template<>
struct traits<QuaternionR4Velocity>
{
	static constexpr int Dim = LIETORCH_QUATERNION_DIM;

	using LieAlg = torch::Tensor;
	using LieGroup = QuaternionR4;
};

template<>
struct traits<Quaternion>
{
	static constexpr int Dim = LIETORCH_QUATERNION_DIM;
	static constexpr int ActDim = LIETORCH_POSITION_DIM;

	using Tangent = AngularVelocity;
	using Vector = torch::Tensor;
	using DataType = torch::Tensor;
};

template<>
struct traits<AngularVelocity>
{
	static constexpr int Dim = LIETORCH_ANGULAR_VELOCITY_DIM;

	using LieAlg = torch::Tensor;
	using LieGroup = Quaternion;
};

}
namespace quaternion_operations
{
using Tensor = torch::Tensor;

// Quaternion operations
Tensor inverse (const Tensor &q);
Tensor action (const Tensor &q, const Tensor &v);
Tensor composition (const Tensor &q1, const Tensor &q2);
Tensor conjugate (const Tensor &q);
Tensor log (const Tensor &q);
Tensor actionJacobianR4 (const Tensor &q, const Tensor &v);
Tensor actionJacobian (const Tensor &q, const Tensor &v);
Tensor normalize (const Tensor &q);
Tensor imag (const Tensor &q);
Tensor distanceRiemann (const Tensor &q1, const Tensor &q2);
// Tangent operations
Tensor exp (const Tensor &t);

Tensor x (const Tensor &q);
Tensor y (const Tensor &q);
Tensor z (const Tensor &q);
Tensor w (const Tensor &q);

}

class AngularVelocity : public Tangent<AngularVelocity>
{
	using Base = Tangent<AngularVelocity>;

public:
	LIETORCH_INHERIT_CONSTRUCTOR(AngularVelocity)
	LIETORCH_INHERIT_TANGENT_TRAITS

	using Base::coeffs;

	LieAlg generator (int i) const;
	LieAlg hat () const;
	LieGroup exp () const;
	DataType norm () const;
	AngularVelocity scale (const DataType &other) const;
};

class Quaternion : public LieGroup<Quaternion>
{
	using Base = LieGroup<Quaternion>;

public:
	LIETORCH_INHERIT_CONSTRUCTOR(Quaternion)
	LIETORCH_INHERIT_GROUP_TRAITS

	using Base::coeffs;

	Quaternion (float x, float y, float z, float w);

	Quaternion inverse () const;
	AngularVelocity log () const;
	Quaternion compose (const Quaternion &other) const;
	DataType dist (const Quaternion &other, const DataType &weights) const;
	Vector act (const Vector &v) const;
	AngularVelocity differentiate(const Vector &outerGradient, const Vector &v, const OpFcn &op) const;

	torch::Tensor x () const;
	torch::Tensor y () const;
	torch::Tensor z () const;
	torch::Tensor w () const;

	Quaternion conj () const;
};


class QuaternionR4Velocity : public Tangent<QuaternionR4Velocity>
{
	using Base = Tangent<QuaternionR4Velocity>;

public:
	using Base::coeffs;

	LIETORCH_INHERIT_TANGENT_TRAITS
	LIETORCH_INHERIT_CONSTRUCTOR(QuaternionR4Velocity)

	LieAlg generator(int i) const;
	LieAlg hat () const;
	LieGroup exp () const;
	DataType norm () const;
	QuaternionR4Velocity scale (const DataType &other) const;

	friend Base;
};

class QuaternionR4 : public LieGroup<QuaternionR4>
{
	using Base = LieGroup<QuaternionR4>;

protected:

	torch::Tensor imag() const;

	QuaternionR4 (const torch::TensorList &coeffsList);

public:
	LIETORCH_INHERIT_GROUP_TRAITS

	using Base::coeffs;

	QuaternionR4 ();
	QuaternionR4 (const QuaternionR4 &other);
	QuaternionR4 (const DataType &_coeffs);
	QuaternionR4 (float x, float y, float z, float w);

	QuaternionR4 inverse () const;
	Tangent log () const;
	QuaternionR4 compose (const QuaternionR4 &o) const;
	DataType dist (const QuaternionR4 &other, const DataType &weights) const;
	Vector act (const Vector &v) const;
	Tangent differentiate (const Vector &outerGradient, const Vector &v, const OpFcn &op = OpIdentity) const;


	// Quaternion specific functions
	void normalize_ ();
	QuaternionR4 normalized();

	torch::Tensor x () const;
	torch::Tensor y () const;
	torch::Tensor z () const;
	torch::Tensor w () const;

	QuaternionR4 conj () const;
	friend Base;
};

}

#endif // QUATERNION_H
