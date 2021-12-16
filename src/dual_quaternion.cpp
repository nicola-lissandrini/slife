#include "lietorch/dual_quaternion.h"

using namespace lietorch;
using namespace torch;
using namespace std;

namespace dual_quaternion_ops {

static const Tensor pureQuaternionMultiplier = torch::tensor({{0,1,0,0},
												  {0,0,1,0},
												  {0,0,0,1}}, kFloat);

Tensor positionToQuaternion (const Tensor &position) {
	return pureQuaternionMultiplier.mv(position);
}

Tensor translationQuaternion (const Tensor &position, const Tensor &rotation) {
	return quaternion_ops::composition(
					   0.5 * dual_quaternion_ops::positionToQuaternion(position),
					   quaternion_ops::conjugate(rotation));
}

}

DualQuaternion::DualQuaternion (const Position3 &_translation, const Quaternion &_rotation):
{
	setRotation(_rotation);
	setTranslation(_translation);
}

DualQuaternion::DualQuaternion (const DataType &_realPart, const DataType &_dualPart):
	Base(torch::cat({_realPart, _dualPart}))
{
}

Tensor DualQuaternion::realPart () const {
	return coeffs.slice(0, 0, LIETORCH_QUATERNION_DIM);
}

Tensor &DualQuaternion::realPart () {
	return coeffs.slice(0, 0, LIETORCH_QUATERNION_DIM);
}

Tensor DualQuaternion::dualPart () const {
	return coeffs.slice(0, LIETORCH_QUATERNION_DIM);
}

Tensor &DualQuaternion::dualPart () {
	return coeffs.slice(0, LIETORCH_QUATERNION_DIM);
}

Quaternion DualQuaternion::realQuaternion () const {
	return Quaternion(realPart());
}

Quaternion DualQuaternion::dualQuaternion() const {
	return Quaternion(dualPart());
}

Quaternion DualQuaternion::rotation () const {
	return Quaternion(realPart());
}

void DualQuaternion::setRotation (const Quaternion &_rotation) {
	realPart() = _rotation.coeffs;

}

Position DualQuaternion::translation () const {
	return Quaternion (2 * dualPart()) * rotation().inverse();
}

void DualQuaternion::setTranslation (const Position &_position) {
	dualPart() = dual_quaternion_ops::translationQuaternion (_position, realPart());
}

DualQuaternion DualQuaternion::inverse () const {
	return DualQuaternion (quaternion_ops::conjugate(realPart()),
					   quaternion_ops::conjugate(dualPart()));
}

