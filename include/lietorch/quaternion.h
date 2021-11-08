#ifndef QUATERNION_H
#define QUATERNION_H

#include "rn.h"

namespace lietorch {

#define LIETORCH_QUATERNION_DIM 4

class UnitQuaternionR4;
class QuaternionR4Velocity;

namespace internal {

template<>
struct traits<UnitQuaternionR4>
{
	static constexpr int Dim = LIETORCH_QUATERNION_DIM;

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

	// Quaternion specific functions
	void normalize_ ();
	UnitQuaternionR4 normalized();

	torch::Tensor x () const;
	torch::Tensor y () const;
	torch::Tensor z () const;
	torch::Tensor w () const;

	UnitQuaternionR4 conj () const;
	friend Base;
};

torch::Tensor UnitQuaternionR4::imag () const {
	return coeffs.index({torch::indexing::Slice(0,3)});
}

UnitQuaternionR4::UnitQuaternionR4():
	Base({0,0,0,1})
{}

UnitQuaternionR4::UnitQuaternionR4(const UnitQuaternionR4 &other):
	Base(other.coeffs)
{}

UnitQuaternionR4::UnitQuaternionR4(const UnitQuaternionR4::DataType &_coeffs):
	Base(_coeffs)
{
	normalize_();
}

UnitQuaternionR4::UnitQuaternionR4(const torch::TensorList &coeffsList) {
	assert (coeffsList.size() == Dim && "Incompatible initializer list size");

	coeffs = torch::cat(coeffsList);
	normalize_ ();
}

UnitQuaternionR4::UnitQuaternionR4(float x, float y, float z, float w)
{
	coeffs = torch::tensor({x, y, z, w}, torch::kFloat);
	normalize_ ();
}

UnitQuaternionR4 UnitQuaternionR4::inverse () const {
	return conj ();
}

UnitQuaternionR4::Tangent UnitQuaternionR4::log() const {
	return Tangent (coeffs);
}

UnitQuaternionR4 UnitQuaternionR4::compose(const UnitQuaternionR4 &o) const {
	return UnitQuaternionR4 ({w()*o.x() + x()*o.w() + y()*o.z() - z()*o.y(),
						 w()*o.y() - x()*o.z() + y()*o.w() + z()*o.x(),
						 w()*o.z() + x()*o.y() - y()*o.x() + z()*o.w(),
						 w()*o.w() - x()*o.x() - y()*o.y() - z()*o.z()});
}

UnitQuaternionR4::Vector UnitQuaternionR4::act (const LieGroup::Vector &v) const
{
	torch::Tensor imagPart = imag ();

	return v * (w().pow(2) - imagPart.pow(2).sum()) +
		  2 * v.matmul(imagPart).unsqueeze(1) * imagPart.unsqueeze(0) +
		  2 * w() * imagPart.expand_as(v).cross (v, 1);
}

void UnitQuaternionR4::normalize_() {
	coeffs = coeffs / coeffs.norm();
}
UnitQuaternionR4 UnitQuaternionR4::normalized () {
	return UnitQuaternionR4 (coeffs / coeffs.norm());
}

torch::Tensor UnitQuaternionR4::x() const {
	return coeffs[0].unsqueeze(0);
}

torch::Tensor UnitQuaternionR4::y() const {
	return coeffs[1].unsqueeze(0);
}

torch::Tensor UnitQuaternionR4::z() const {
	return coeffs[2].unsqueeze(0);
}

torch::Tensor UnitQuaternionR4::w() const {
	return coeffs[3].unsqueeze(0);
}

UnitQuaternionR4 UnitQuaternionR4::conj() const {
	return UnitQuaternionR4({-x(),
						-y(),
						-z(),
						 w()});
}

QuaternionR4Velocity::LieAlg QuaternionR4Velocity::generator (int i) const {
	torch::Tensor gen = torch::zeros({Dim}, torch::kFloat);
	gen[i] = 1;
	return gen;
}

QuaternionR4Velocity::LieGroup QuaternionR4Velocity::exp () const {
	return LieGroup (coeffs);
}

QuaternionR4Velocity QuaternionR4Velocity::scale(const QuaternionR4Velocity::DataType &other) const {
	return QuaternionR4Velocity (otehr)
}

}

#endif // QUATERNION_H
