#include "lietorch/pose.h"

using namespace lietorch;
using namespace torch;
using namespace std;

// Instantiate implemented templates
template
class PoseBase<Position3, QuaternionR4>;
template
class TwistBase<Position3, QuaternionR4>;

template<class Translation, class Rotation>
Translation PoseBase<Translation, Rotation>::translation () const {
	const int tDim = Translation::Dim;
	return Translation (coeffs.slice(0, 0, tDim));
}


template<class Translation, class Rotation>
Rotation PoseBase<Translation, Rotation>::rotation () const {
	const int rDim = Rotation::Dim;
	const int tDim = Translation::Dim;
	return Rotation (coeffs.slice(0, tDim, tDim + rDim));
}

template<class Translation, class Rotation>
PoseBase<Translation, Rotation>::PoseBase (const Translation &translation, const Rotation &rotation):
	Base(torch::cat({translation.coeffs, rotation.coeffs}))
{
}

template<class Translation, class Rotation>
PoseBase<Translation, Rotation> PoseBase<Translation, Rotation>::inverse () const
{
	Rotation inv = rotation().inverse();
	return PoseBase(inv * (translation().inverse()), inv);
}

template<class Translation, class Rotation>
typename PoseBase<Translation,Rotation>::Tangent PoseBase<Translation,Rotation>::log () const {
	return Tangent (translation().log(), rotation().log());
}

// Composition is different according to each specialization
// Pose3R4
template<>
PoseBase<Position3, QuaternionR4> PoseBase<Position3, QuaternionR4>::compose (const PoseBase &other) const {
	return PoseBase (translation() * other.translation(), rotation() * other.rotation());
}

template<class Translation, class Rotation>
typename PoseBase<Translation,Rotation>::DataType
PoseBase<Translation,Rotation>::dist(const PoseBase &other, const DataType &weights) const {
	assert ((weights.dim() == 1 && weights.size(0) == 2) && "Rn must be weighted by a 1d vector of length 2");

	return translation().dist(other.translation(), weights[0].unsqueeze(0)) + rotation().dist(other.rotation(), weights[1].unsqueeze(0));
}

template<class Translation, class Rotation>
typename PoseBase<Translation, Rotation>::Vector
PoseBase<Translation, Rotation>:: PoseBase::act (const Vector &v) const {
	return rotation() * v + translation().coeffs;
}

template<class Translation, class Rotation>
typename PoseBase<Translation, Rotation>::Tangent
PoseBase<Translation, Rotation>::differentiate (const Vector &outerGradient, const Vector &v) const {
	return Tangent (translation().differentiate (outerGradient, v),
				 rotation().differentiate (outerGradient, v));
}

template<class Translation, class Rotation>
TwistBase<Translation, Rotation>::TwistBase(const LinearVelocity &linear, const AngularVelocity &angular):
	Base(torch::cat ({linear.coeffs, angular.coeffs}))
{}

template<class Translation, class Rotation>
typename TwistBase<Translation, Rotation>::LinearVelocity
TwistBase<Translation, Rotation>::linear () const {
	// + sign fixes ODR violation issue prior to C++ 17
	return coeffs.slice (0, 0, +LinearVelocity::Dim);
}

template<class Translation, class Rotation>
typename TwistBase<Translation, Rotation>::AngularVelocity
TwistBase<Translation, Rotation>::angular () const {
	// + sign fixes ODR violation issue prior to C++ 17
	return coeffs.slice (0, +LinearVelocity::Dim, +LinearVelocity::Dim + AngularVelocity::Dim);
}

template<class Translation, class Rotation>
typename TwistBase<Translation, Rotation>::LieGroup
TwistBase<Translation, Rotation>::exp () const {
	return LieGroup (linear().exp(), angular().exp ());
}

template<class Translation, class Rotation>
TwistBase<Translation, Rotation> TwistBase<Translation, Rotation>::scale(const TwistBase::DataType &other) const
{
	assert (other.sizes().size() == 1 && other.size(0) == 2 && "Scaling tensor must be 1D and with exactly two elemenents");

	return TwistBase (linear() * other[0], angular() * other[1]);
}
