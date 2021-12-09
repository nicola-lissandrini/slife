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
	static constexpr int ActDim = LIETORCH_POSITION_DIM;


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

public:
	using Base::coeffs;

	LIETORCH_INHERIT_GROUP_TRAITS

	PoseBase (const Translation &_position = Translation (), const Rotation &_orientation = Rotation ());

	PoseBase inverse () const;
	Tangent log () const;
	PoseBase compose (const PoseBase &other) const;
	DataType dist (const PoseBase &other, const DataType &weights) const;
	Vector act (const Vector &v) const;
	Tangent differentiate (const Vector &outerGradient, const Vector &v, const std::function<torch::Tensor(torch::Tensor)> &op = std::function<torch::Tensor(torch::Tensor)> ()) const;

	Translation translation () const;
	Rotation rotation () const;
};

// Actual Definitions
//using Position2 = Rn<2>;
using Position3 = Rn<3>;

//using Velocity2 = VelocityRn<2>;
using Velocity3 = VelocityRn<3>;

using Pose3R4 = PoseBase<Position3, QuaternionR4>;
using Pose = PoseBase<Position3, Quaternion>;
using Twist3R4 = TwistBase<Velocity3, QuaternionR4Velocity>;

using Position = Position3;
using Velocity = Velocity3;




}




#endif // POSE_H
