#include "lietorch/quaternion.h"
#include "../../sparcsnode/include/sparcsnode/utils.h"

using namespace lietorch;
using namespace torch;
using namespace std;

/****
 * Quaternion operations on tensors
 * **/

namespace lietorch::quaternion_operations {

static const Tensor jacobianMultiplier = 2*torch::tensor({{{{ 1, 0, 0}, { 0,-1, 0}, { 0, 0,-1}},
											{{ 0, 1, 0}, { 1, 0, 0}, { 0, 0, 0}},
											{{ 0, 0, 1}, { 0, 0, 0}, { 1, 0, 0}},
											{{ 0, 0, 0}, { 0, 0,-1}, { 0, 1, 0}}},
										    {{{ 0, 1, 0}, { 1, 0, 0}, { 0, 0, 0}},
											{{-1, 0, 0}, { 0, 1, 0}, { 0, 0,-1}},
											{{ 0, 0, 0}, { 0, 0, 1}, { 0, 1, 0}},
											{{ 0, 0, 1}, { 0, 0, 0}, {-1, 0, 0}}},
										    {{{ 0, 0, 1}, { 0, 0, 0}, { 1, 0, 0}},
											{{ 0, 0, 0}, { 0, 0, 1}, { 0, 1, 0}},
											{{-1, 0, 0}, { 0,-1, 0}, { 0, 0, 1}},
											{{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 0}}},
										    {{{ 0, 0, 0}, { 0, 0,-1}, { 0, 1, 0}},
											{{ 0, 0, 1}, { 0, 0, 0}, {-1, 0, 0}},
											{{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 0}},
											{{ 1, 0, 0}, { 0, 1, 0}, { 0, 0, 1}}}}, kFloat);

static const Tensor compositionMultiplier = torch::tensor({{{ 0, 0, 0, 1}, { 0, 0, 1, 0}, { 0,-1, 0, 0}, {-1, 0, 0, 0}},
											    {{ 0, 0,-1, 0}, { 0, 0, 0, 1}, { 1, 0, 0, 0}, { 0,-1, 0, 0}},
											    {{ 0, 1, 0, 0}, {-1, 0, 0, 0}, { 0, 0, 0, 1}, { 0, 0,-1, 0}},
											    {{ 1, 0, 0, 0}, { 0, 1, 0, 0}, { 0, 0, 1, 0}, { 0, 0, 0, 1}}}, kFloat);

static const Tensor conjugateMultiplier = torch::tensor({{-1, 0, 0, 0}, { 0,-1, 0, 0}, { 0, 0,-1, 0}, { 0, 0, 0, 1}}, kFloat);

static const Tensor skewMultiplier = torch::tensor({{{ 0, 0, 0}, { 0, 0,-1}, { 0, 1, 0}},
										  {{ 0, 0, 1}, { 0, 0, 0}, {-1, 0, 0}},
										  {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 0}}}, kFloat);

static const Tensor rotationMultiplier = torch::tensor({{{ 1, 0, 0, 0}, { 0,-1, 0, 0}, { 0, 0,-1, 0}, { 0, 0, 0, 1}},
											 {{ 0, 2, 0, 0}, { 0, 0, 0, 0}, { 0, 0, 0, 0}, { 0, 0,-2, 0}},
											 {{ 0, 0, 2, 0}, { 0, 0, 0, 0}, { 0, 0, 0, 0}, { 0, 2, 0, 0}},
											 {{ 0, 2, 0, 0}, { 0, 0, 0, 0}, { 0, 0, 0, 0}, { 0, 0, 2, 0}},
											 {{-1, 0, 0, 0}, { 0, 1, 0, 0}, { 0, 0,-1, 0}, { 0, 0, 0, 1}},
											 {{ 0, 0, 0, 0}, { 0, 0, 2, 0}, { 0, 0, 0, 0}, {-2, 0, 0, 0}},
											 {{ 0, 0, 2, 0}, { 0, 0, 0, 0}, { 0, 0, 0, 0}, { 0,-2, 0, 0}},
											 {{ 0, 0, 0, 0}, { 0, 0, 2, 0}, { 0, 0, 0, 0}, { 2, 0, 0, 0}},
											 {{-1, 0, 0, 0}, { 0,-1, 0, 0}, { 0, 0, 1, 0}, { 0, 0, 0, 1}}}, kFloat);

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
	return (compositionMultiplier * q2.unsqueeze(1).unsqueeze(1)).sum(0).matmul(q1);
}

Tensor conjugate (const Tensor &q) {
	return conjugateMultiplier.matmul(q.unsqueeze(1)).squeeze();
}

Tensor actionJacobian (const Tensor &q, const Tensor &v) {
	return -q.unsqueeze(0).unsqueeze(1).matmul(rotationMultiplier)
			.matmul(q.unsqueeze(1)
				    .unsqueeze(0))
			.squeeze().reshape({3,3})
			.unsqueeze(0)
			.matmul((skewMultiplier.unsqueeze(3) *
				    v.t().unsqueeze(1).unsqueeze(1)
				    ).sum(0)
					.permute({2,0,1}));
}

Tensor actionJacobianR4 (const Tensor &q, const Tensor &v) {
	return (jacobianMultiplier * q.unsqueeze(1).unsqueeze(1))
			.sum(1).matmul(v.t())
			.permute({2,0,1})
			.transpose(1,2);
}

Tensor normalize (const Tensor &q) {
	return q / q.norm(2,0); // Todo: check dim
}

Tensor distanceRiemann (const Tensor &q1, const Tensor &q2) {
	return quaternion_operations::log(composition (quaternion_operations::inverse(q1), q2)).norm ();
}

Tensor inverse(const Tensor &q) {
	return quaternion_operations::conjugate(q);
}

Tensor log (const Tensor &q) {
	Tensor imagPart = quaternion_operations::imag(q);
	Tensor imagPartNorm = imagPart.norm();
	return 2 * imagPart / imagPartNorm * torch::acos(w(q));
}

Tensor exp (const Tensor &t) {
	Tensor theta = t.norm(2,0);

	if (theta.item().toFloat() < 1e-10)
		return torch::cat ({0 * t, theta.cos().unsqueeze(0) });
	else
		return torch::cat ({t / theta * (theta / 2).sin(), (theta / 2).cos().unsqueeze(0)});
}


} // namespace quaternion_operations


/****
 * Quaternion seen as proper Quaternion Lie Group
 * **/
Quaternion::Quaternion (float x, float y, float z, float w):
	Quaternion(torch::tensor({x, y, z, w}, kFloat))
{}

Quaternion Quaternion::inverse () const {
	return quaternion_operations::inverse(coeffs);
}

AngularVelocity Quaternion::log() const {
	return quaternion_operations::log(coeffs);
}

Quaternion Quaternion::compose(const Quaternion &other) const {
	return quaternion_operations::composition(coeffs, other.coeffs);
}

Quaternion::Vector Quaternion::act(const Vector &v) const {
	return quaternion_operations::action(coeffs, v);
}

Quaternion::DataType Quaternion::dist (const Quaternion &other, const DataType &weights) const {
	return quaternion_operations::distanceRiemann (coeffs, other.coeffs);
}

AngularVelocity Quaternion::differentiate(const Vector &outerGradient, const Vector &v, const OpFcn &op, const boost::optional<Tensor &> &jacobian) const
{
	Tensor actionJacobian = quaternion_operations::actionJacobian (coeffs, v);
	Tensor gradientTensor = outerGradient.unsqueeze(1).matmul (actionJacobian).squeeze();
	if (jacobian)
		*jacobian = gradientTensor;
	return AngularVelocity (op (gradientTensor));
}

Quaternion Quaternion::conj() const {
	return quaternion_operations::conjugate(coeffs);
}

torch::Tensor Quaternion::x() const {
	return quaternion_operations::x (coeffs);
}

torch::Tensor Quaternion::y() const {
	return quaternion_operations::y (coeffs);
}

torch::Tensor Quaternion::z() const {
	return quaternion_operations::z (coeffs);
}

torch::Tensor Quaternion::w() const {
	return quaternion_operations::w (coeffs);
}

Quaternion AngularVelocity::exp() const {
	return quaternion_operations::exp(coeffs);
}

AngularVelocity::DataType AngularVelocity::norm() const {
	return coeffs.norm (2, 0);
}

AngularVelocity AngularVelocity::scale(const DataType &other) const {
	assert ((other.dim() == 1 && other.sizes()[0] == 1) || other.dim() == 0 && "QuaternionR4Velocity can only scale by a scalar");

	return coeffs * other;
}

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


QuaternionR4Velocity QuaternionR4::differentiate (const Vector &outerGradient, const Vector &v, const OpFcn &op, const boost::optional<torch::Tensor &> &jacobian) const
{
	Tensor actionJacobian = quaternion_operations::actionJacobianR4 (coeffs, v);
	Tensor gradientTensor = outerGradient.unsqueeze (1).matmul (actionJacobian).squeeze ();
	if (jacobian)
		*jacobian = gradientTensor;
	return QuaternionR4Velocity(op (gradientTensor));
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
