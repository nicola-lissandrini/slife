#include "lietorch/quaternion.h"

using namespace lietorch;
using namespace torch;

Tensor UnitQuaternionR4::imag () const {
	return coeffs.slice(0,3);
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

UnitQuaternionR4::UnitQuaternionR4(const torch::TensorList &coeffsList):
	UnitQuaternionR4(torch::cat(coeffsList))
{
}

UnitQuaternionR4::UnitQuaternionR4 (float x, float y, float z, float w):
	UnitQuaternionR4 (torch::tensor({x, y, z, w}, torch::kFloat))
{
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

UnitQuaternionR4::Tangent UnitQuaternionR4::differentiate (const Vector &outerGradient, const Vector &v) const
{
	// Differentiate rotation action R*v
	Jacobian jacobianCopy = jacobian;
	jacobianCopy[0, 0] =  2*v[0]*x() + 2*v[1]*y() + 2*v[2]*z();
	jacobianCopy[1, 0] =  2*v[0]*y() - 2*v[1]*x() - 2*v[2]*w();
	jacobianCopy[2, 0] =  2*v[0]*z() + 2*v[1]*w() - 2*v[2]*x();
	jacobianCopy[0, 1] = -2*v[0]*y() + 2*v[1]*x() + 2*v[2]*w();
	jacobianCopy[1, 1] =  2*v[0]*x() + 2*v[1]*y() + 2*v[2]*z();
	jacobianCopy[2, 1] = -2*v[0]*w() + 2*v[1]*z() - 2*v[2]*y();
	jacobianCopy[0, 2] = -2*v[0]*z() - 2*v[1]*w() + 2*v[2]*x();
	jacobianCopy[1, 2] =  2*v[0]*w() - 2*v[1]*z() + 2*v[2]*y();
	jacobianCopy[2, 2] =  2*v[0]*x() + 2*v[1]*y() + 2*v[2]*z();
	jacobianCopy[0, 3] =  2*v[0]*w() - 2*v[1]*z() + 2*v[2]*y();
	jacobianCopy[1, 3] =  2*v[0]*z() + 2*v[1]*w() - 2*v[2]*x();
	jacobianCopy[2, 3] = -2*v[0]*y() + 2*v[1]*x() + 2*v[2]*w();

	return Tangent (outerGradient.unsqueeze (0).mm (jacobianCopy).squeeze ());
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

QuaternionR4Velocity QuaternionR4Velocity::scale(const QuaternionR4Velocity::DataType &other) const
{
	assert (other.sizes().size() == 1 && other.sizes()[0] == 1 && "VelocityRn can only scale by a scalar");

	return QuaternionR4Velocity (coeffs * other);
}
