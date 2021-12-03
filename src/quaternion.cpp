#include "lietorch/quaternion.h"

using namespace lietorch;
using namespace torch;
using namespace std;

/****
 * Quaternion operations on tensors
 * **/

namespace lietorch::quaternion_operations {

// Single element or vector of selected component from the tensor
Tensor x (const Tensor &q) {
	return (q.dim() == 1? q.unsqueeze(0) : q).slice(1,0,1).squeeze(1);
}

Tensor y (const Tensor &q) {
	return (q.dim() == 1? q.unsqueeze(0) : q).slice(1,1,2).squeeze(1);
}

Tensor z (const Tensor &q) {
	return (q.dim() == 1? q.unsqueeze(0) : q).slice(1,2,3).squeeze(1);
}

Tensor w (const Tensor &q) {
	return (q.dim() == 1? q.unsqueeze(0) : q).slice(1,3,4).squeeze(1);
}

Tensor imag (const Tensor &q) {
	return q.slice(q.dim() > 1? 1 : 0, 0, 3);
}

Tensor action (const Tensor &q, const Tensor &v)
{
	torch::Tensor imagPart = quaternion_operations::imag (q);

	if (v.sizes().size() == 1)
		v.unsqueeze_(0);

	return (v * (w(q).pow(2) - imagPart.pow(2).sum()) +
		  2 * v.matmul(imagPart).unsqueeze(1) * imagPart.unsqueeze(0) +
		  2 * w(q) * imagPart.expand_as(v).cross (v, 1)).squeeze();
}

Tensor composition (const Tensor &q1, const Tensor &q2) {
	return torch::cat({w(q1)*x(q2) + x(q1)*w(q2) + y(q1)*z(q2) - z(q1)*y(q2),
				    w(q1)*y(q2) - x(q1)*z(q2) + y(q1)*w(q2) + z(q1)*x(q2),
				    w(q1)*z(q2) + x(q1)*y(q2) - y(q1)*x(q2) + z(q1)*w(q2),
				    w(q1)*w(q2) - x(q1)*x(q2) - y(q1)*y(q2) - z(q1)*z(q2)});
}

Tensor conjugate (const Tensor &q) {
	return torch::cat({-x(q),
				    -y(q),
				    -z(q),
					w(q)});
}

Tensor actionJacobian (const Tensor &q, const Tensor &v) {
	Tensor j0 = torch::cat ({ 2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q),
						 2*v[0]*y(q) - 2*v[1]*x(q) - 2*v[2]*w(q),
						 2*v[0]*z(q) + 2*v[1]*w(q) - 2*v[2]*x(q)});
	Tensor j1 = torch::cat ({-2*v[0]*y(q) + 2*v[1]*x(q) + 2*v[2]*w(q),
						 2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q),
						-2*v[0]*w(q) + 2*v[1]*z(q) - 2*v[2]*y(q)});
	Tensor j2 = torch::cat ({-2*v[0]*z(q) - 2*v[1]*w(q) + 2*v[2]*x(q),
						 2*v[0]*w(q) - 2*v[1]*z(q) + 2*v[2]*y(q),
						 2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q)});
	Tensor j3 = torch::cat ({ 2*v[0]*w(q) - 2*v[1]*z(q) + 2*v[2]*y(q),
						 2*v[0]*z(q) + 2*v[1]*w(q) - 2*v[2]*x(q),
						-2*v[0]*y(q) + 2*v[1]*x(q) + 2*v[2]*w(q)});
	return torch::stack({j0,j1,j2,j3}).t();
}

Tensor normalize (const Tensor &q) {
	return q / q.norm(2,0); // Todo: check dim
}


// Tensor log (const Tensor &q) { ... todo }


} // namespace quaternion_operations

/****
 * Quaternion seen as R4 Lie group
 * **/


Tensor QuaternionR4::imag () const {
	return coeffs.slice(0,0,3);
}

QuaternionR4::QuaternionR4():
	Base({0,0,0,1})
{}

QuaternionR4::QuaternionR4(const QuaternionR4 &other):
	Base(other.coeffs)
{}

QuaternionR4::QuaternionR4(const QuaternionR4::DataType &_coeffs):
	Base(_coeffs)
{
}

QuaternionR4::QuaternionR4(const torch::TensorList &coeffsList):
	QuaternionR4(torch::cat(coeffsList))
{
}

QuaternionR4::QuaternionR4 (float x, float y, float z, float w):
	QuaternionR4 (torch::tensor({x, y, z, w}, torch::kFloat))
{
}

QuaternionR4 QuaternionR4::inverse () const {
	// R4 inverse
	return QuaternionR4(-coeffs);
}

QuaternionR4::Tangent QuaternionR4::log() const {
	return Tangent (coeffs);
}

QuaternionR4 QuaternionR4::compose(const QuaternionR4 &o) const {
	// R4 composition
	return QuaternionR4(coeffs + o.coeffs).normalized();
}

QuaternionR4::Vector QuaternionR4::act(const Vector &v) const {
	// Quaternion action
	return quaternion_operations::action(coeffs, v);
}

QuaternionR4::DataType QuaternionR4::dist(const QuaternionR4 &other, const DataType &weights) const {
	assert ((weights.dim() == 1 && weights.size(0) == 1) && "QuaternionR4 must be weighted by a 1d scalar");

	return (coeffs - other.coeffs).norm () * weights;
}


QuaternionR4::Tangent QuaternionR4::differentiate (const Vector &outerGradient, const Vector &v) const
{
	Tensor jacobian = quaternion_operations::actionJacobian (coeffs, v);
	return Tangent (outerGradient.unsqueeze (0).mm (jacobian).squeeze ());
}

void QuaternionR4::normalize_() {
	coeffs = quaternion_operations::normalize (coeffs);
}
QuaternionR4 QuaternionR4::normalized () {
	return QuaternionR4 (quaternion_operations::normalize (coeffs));
}

torch::Tensor QuaternionR4::x() const {
	return quaternion_operations::x (coeffs);
}

torch::Tensor QuaternionR4::y() const {
	return quaternion_operations::y (coeffs);
}

torch::Tensor QuaternionR4::z() const {
	return quaternion_operations::z (coeffs);
}

torch::Tensor QuaternionR4::w() const {
	return quaternion_operations::w (coeffs);
}

QuaternionR4Velocity::LieAlg QuaternionR4Velocity::generator (int i) const {
	torch::Tensor gen = torch::zeros({Dim}, torch::kFloat);
	gen[i] = 1;
	return gen;
}

QuaternionR4Velocity::LieGroup QuaternionR4Velocity::exp () const {
	// R4 exp
	return LieGroup (coeffs);
}

QuaternionR4Velocity::DataType QuaternionR4Velocity::norm() const {
	return coeffs.norm (2,0);
}

QuaternionR4Velocity QuaternionR4Velocity::scale (const QuaternionR4Velocity::DataType &other) const
{
	assert ((other.sizes().size() == 1 && other.sizes()[0] == 1) || other.sizes().size() == 0 && "QuaternionR4Velocity can only scale by a scalar");

	return QuaternionR4Velocity (coeffs * other);
}
