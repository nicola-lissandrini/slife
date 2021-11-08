#ifndef QUATERNION_H
#define QUATERNION_H

#include "rn.h"

namespace lietorch {

#define LIETORCH_QUATERNION_DIM 4
#define LIETORCH_POSITION_DIM 3

class UnitQuaternionR4;
class QuaternionR4Velocity;

namespace internal {

template<>
struct traits<UnitQuaternionR4>
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
	using LieGroup = UnitQuaternionR4;
};

}

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
	QuaternionR4Velocity scale (const DataType &other) const;

	friend Base;
};

class UnitQuaternionR4 : public LieGroup<UnitQuaternionR4>
{
	using Base = LieGroup<UnitQuaternionR4>;

protected:

	torch::Tensor imag() const;

	UnitQuaternionR4 (const torch::TensorList &coeffsList);

public:
	LIETORCH_INHERIT_GROUP_TRAITS

	using Base::coeffs;

	UnitQuaternionR4 ();
	UnitQuaternionR4 (const UnitQuaternionR4 &other);
	UnitQuaternionR4 (const DataType &_coeffs);
	UnitQuaternionR4 (float x, float y, float z, float w);

	UnitQuaternionR4 inverse () const;
	Tangent log () const;
	UnitQuaternionR4 compose (const UnitQuaternionR4 &o) const;
	Vector act (const Vector &v) const;
	Tangent differentiate (const Vector &outerGradient, const Vector &v) const;


	// Quaternion specific functions
	void normalize_ ();
	UnitQuaternionR4 normalized();

	torch::Tensor x () const;
	torch::Tensor y () const;
	torch::Tensor z () const;
	torch::Tensor w () const;

	UnitQuaternionR4 conj () const;
	friend Base;

private:
	using Base::jacobian;
};

}

#endif // QUATERNION_H
