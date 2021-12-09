#ifndef RN_H
#define RN_H

#include "liegroup.h"


namespace lietorch {

template<int _N>
class Rn;

template<int _N>
class VelocityRn;

namespace internal {

template<int _N>
struct traits<Rn<_N>>
{
	static constexpr int Dim = _N;
	static constexpr int ActDim = _N;

	using Tangent = VelocityRn<_N>;
	using Vector = torch::Tensor;
	using DataType = torch::Tensor;
};

template<int _N>
struct traits<VelocityRn<_N>>
{
	static constexpr int Dim = _N;

	using LieAlg = torch::Tensor;
	using LieGroup = Rn<_N>;
};

}

template<int _N>
class VelocityRn : public Tangent<VelocityRn<_N>>
{
	using Base = Tangent<VelocityRn<_N>>;

public:
	using Base::coeffs;

	LIETORCH_INHERIT_TANGENT_TRAITS
	LIETORCH_INHERIT_CONSTRUCTOR(VelocityRn)

	LieAlg generator(int i) const;
	LieAlg hat () const;
	LieGroup exp () const;
	DataType norm () const;
	VelocityRn scale(const DataType &other) const;
};

template<int _N>
class Rn : public LieGroup<Rn<_N>>
{
	static_assert(_N > 0, "N must be greater than 0");

	using Base = LieGroup<Rn<_N>>;

public:
	using Base::coeffs;

	LIETORCH_INHERIT_GROUP_TRAITS
	LIETORCH_INHERIT_CONSTRUCTOR(Rn)

	Rn inverse () const;
	Tangent log () const;
	Rn compose (const Rn &other) const;
	DataType dist (const Rn &other, const DataType &weights) const;
	Vector act (const Vector &v) const;
	Tangent differentiate (const DataType &outerGradient,
					   const Vector &v,
					   const OpFcn &op = OpIdentity,
					   const boost::optional<torch::Tensor &> &jacobian = boost::none) const;

};

// Implementation
template<int _N>
Rn<_N> Rn<_N>::inverse () const {
	return Rn<_N> (-coeffs);
}

template<int _N>
typename Rn<_N>::Tangent
Rn<_N>::log() const {
	return VelocityRn<_N> (coeffs);
}

template<int _N>
Rn<_N>  Rn<_N>::compose (const Rn<_N> &other) const {
	return Rn(coeffs + other.coeffs);
}

template<int _N>
typename Rn<_N>::DataType Rn<_N>::dist(const Rn<_N> &other, const DataType &weights) const {
	assert ((weights.dim() == 1 && weights.size(0) == 1) && "Rn must be weighted by a 1d scalar");

	return (coeffs - other.coeffs).norm();
}

template<int _N>
typename Rn<_N>::Vector Rn<_N>::act (const Vector &v) const {
	return coeffs + v;
}

template<int _N>
typename Rn<_N>::Tangent Rn<_N>::differentiate (const DataType &outerGradient, const Vector &v, const OpFcn &op, const boost::optional<torch::Tensor &> &jacobian) const {
	// Jacobian is the identity
	if (jacobian)
		*jacobian = outerGradient;
	return op (outerGradient);
}

template<int _N>
typename VelocityRn<_N>::LieAlg VelocityRn<_N>::generator (int i) const {
	torch::Tensor gen = torch::zeros({Dim}, torch::kFloat);
	gen[i] = 1;
	return gen;
}

template<int _N>
typename VelocityRn<_N>::LieGroup VelocityRn<_N>::exp () const {
	return LieGroup (coeffs);
}

template<int _N>
typename VelocityRn<_N>::DataType VelocityRn<_N>::norm() const {
	return coeffs.norm();
}

template<int _N>
VelocityRn<_N> VelocityRn<_N>::scale(const DataType &other) const
{
	assert (((other.dim() == 1 && other.size(0) == 1) || other.dim() == 0) && "VelocityRn can only scale by a scalar");

	return VelocityRn<_N> (coeffs * other);
}


}
#endif // RN_H














