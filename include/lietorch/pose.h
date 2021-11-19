#ifndef POSE_H
#define POSE_H

#include "rn.h"
#include "quaternion.h"

//#include <manif/impl/se3/SE3.h>

namespace lietorch {


template<class Translation, class Rotation>
class TwistBase;
template<class Translation, class Rotation>
class PoseBase;

namespace internal {

template<class Translation, class Rotation>
struct traits<PoseBase<Translation, Rotation>>
{
	static constexpr int Dim = Translation::Dim + Rotation::Dim;

	using Tangent = TwistBase<Translation, Rotation>;
	using Vector = torch::Tensor;
	using DataType = torch::Tensor;
};

template<class Translation, class Rotation>
struct traits<TwistBase<Translation, Rotation>>
{
	static constexpr int Dim = Translation::Dim + Rotation::Dim;

	using LieAlg = torch::Tensor;
	using LieGroup = PoseBase<Translation, Rotation>;
	using DataType = torch::Tensor;
};

}

template<class Translation, class Rotation>
class TwistBase : public Tangent<TwistBase<Translation,Rotation>>
{
	using Base = Tangent<TwistBase<Translation,Rotation>>;

	using LinearVelocity = typename Translation::Tangent;
	using AngularVelocity = typename Rotation::Tangent;

public:
	using Base::coeffs;
	LIETORCH_INHERIT_TANGENT_TRAITS

	TwistBase (const LinearVelocity &linear = LinearVelocity(), const AngularVelocity &angular = AngularVelocity());

	LieAlg generator(int i) const;
	LieAlg hat () const;
	LieGroup exp() const;
	TwistBase scale (const DataType &other) const;

	LinearVelocity linear () const;
	AngularVelocity angular () const;
};

template<class Translation, class Rotation>
class PoseBase : public LieGroup<PoseBase<Translation,Rotation>>
{
	using Base = LieGroup<PoseBase<Translation,Rotation>>;
	using Base::coeffs;

public:
	LIETORCH_INHERIT_GROUP_TRAITS

	PoseBase (const Translation &_position = Translation (), const Rotation &_orientation = Rotation ());

	PoseBase inverse () const;
	Tangent log () const;
	PoseBase compose (const PoseBase &other) const;
	Vector act (const Vector &v) const;
	Tangent differentiate (const Vector &outerGradient, const Vector &v) const;

	Translation translation () const;
	Rotation rotation () const;
};

// Actual Definitions
using Position2 = Rn<2>;
using Position3 = Rn<3>;
using Rotation = UnitQuaternionR4;

using Velocity2 = VelocityRn<2>;
using Velocity3 = VelocityRn<3>;

using Pose3R4 = PoseBase<Position3, UnitQuaternionR4>;
// using Pose3H = PoseBase<Position3, QuaternionH>; TODO
using Twist3R4 = TwistBase<Velocity3, QuaternionR4Velocity>;

using Position = Position3;
using Velocity = Velocity3;
using Pose = Pose3R4;
using Twist = Twist3R4;



template<class Translation, class Rotation>
Translation PoseBase<Translation, Rotation>::translation () const {
	const int tDim = Translation::Dim;
	return Translation (coeffs.slice(0, 0, tDim));
}


template<class Translation, class Rotation>
Rotation PoseBase<Translation, Rotation>::rotation () const {
	const int rDim = Rotation::Dim;
	const int tDim = Translation::Dim;
	return coeffs.slice(0, tDim, tDim + rDim);
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
PoseBase<Translation, Rotation> PoseBase<Translation, Rotation>::compose (const PoseBase &other) const {
	return PoseBase (translation() * (rotation() * other.translation()), rotation() * other.rotation());
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
	return coeffs.slice (0, 0, LinearVelocity::Dim);
}

template<class Translation, class Rotation>
typename TwistBase<Translation, Rotation>::AngularVelocity
TwistBase<Translation, Rotation>::angular () const {
	return coeffs.slice (0, LinearVelocity::Dim, LinearVelocity::Dim + AngularVelocity::Dim);
}

template<class Translation, class Rotation>
typename TwistBase<Translation, Rotation>::LieGroup
TwistBase<Translation, Rotation>::exp () const {
	return LieGroup (linear().exp(), angular().exp ());
}

template<class Translation, class Rotation>
TwistBase<Translation, Rotation> TwistBase<Translation, Rotation>::scale(const TwistBase::DataType &other) const
{

}

}




#endif // POSE_H
