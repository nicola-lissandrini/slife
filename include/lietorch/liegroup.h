#ifndef LIEGROUP_H
#define LIEGROUP_H

#include <cxxabi.h>

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include "tangent.h"

namespace lietorch {

namespace internal {

template<class T>
struct traits;

}

#define LIETORCH_INHERIT_CONSTRUCTOR(X) \
	X(): Base () {} \
	X(const DataType &coeffs): Base (coeffs) {} \
	explicit X(const torch::detail::TensorDataContainer &coeffsList): Base (coeffsList) {}\
	X &operator = (const torch::detail::TensorDataContainer &coeffsList) { return Base::operator = (coeffsList);}\
	friend Base;

#define LIETORCH_INHERIT_GROUP_TRAITS         \
	using Jacobian = typename Base::Jacobian; \
	using DataType = typename Base::DataType;\
	using Vector = typename Base::Vector;    \
	using Tangent = typename Base::Tangent;  \
	using Base::Dim;


template<class Derived>
class LieGroup
{
public:
	using DataType = torch::Tensor;
	using Jacobian = torch::Tensor;
	using Tangent  = typename internal::traits<Derived>::Tangent;
	using Vector   = typename internal::traits<Derived>::Vector;

	static constexpr int Dim = internal::traits<Derived>::Dim;
	static constexpr int ActDim = internal::traits<Derived>::ActDim;

protected:
	inline Derived &derived () { return *static_cast<Derived*>(this); }
	inline const Derived &derived () const { return *static_cast<const Derived*>(this); }

	Jacobian jacobian;

private:
	void initJacobian ();

public:
	DataType coeffs;

	LieGroup();
	LieGroup(const DataType &_coeffs);
	explicit LieGroup(const torch::detail::TensorDataContainer &coeffsList);

	Derived &operator = (const LieGroup &other);
	Derived &operator = (const torch::detail::TensorDataContainer &coeffsList);
	Derived inverse () const;
	Tangent log () const;
	Derived compose (const Derived &other) const;
	Vector act (const Vector &v) const;
	Tangent differentiate (const Vector &outerGradient, const Vector &v) const;

	// this * t.exp ()
	Derived plus (const Tangent &t) const;
	// (other.inverse * this).log ()
	Tangent minus (const Derived &other) const;

	bool isApprox (const Derived &other) const;

	// same as plus ()
	Derived operator + (const Tangent &t) const;
	// equivalent to plus + (-t)
	Derived operator - (const Tangent &t) const;
	// same as minus ()
	Tangent operator - (const Derived &t) const;
	// Compose
	Derived operator * (const Derived &other) const;
	// Act
	Vector operator * (const Vector &other) const;
	template<class DerivedOther>
	Vector operator * (const DerivedOther &other) const;

	Derived &setIdentity ();
	static Derived Identity ();
};

template<class Derived>
std::ostream &operator << (std::ostream &os, const LieGroup<Derived> &l) {
	os << abi::__cxa_demangle(typeid(Derived).name(), NULL,NULL,NULL) << "\n" << l.coeffs;

	return os;
}

template<typename Derived>
void LieGroup<Derived>::initJacobian() {
	jacobian = torch::empty ({ActDim, Dim}, torch::kFloat);
}

// Copy
template<typename Derived>
LieGroup<Derived>::LieGroup() {
	initJacobian();
	setIdentity();
}

template<class Derived>
LieGroup<Derived>::LieGroup(const LieGroup::DataType &_coeffs):
	coeffs(_coeffs)
{
	assert (coeffs.sizes().size() == 1 && coeffs.size(0) == Dim && "Incompatible initialization tensor size");

	initJacobian();
}

template<class Derived>
LieGroup<Derived>::LieGroup(const torch::detail::TensorDataContainer &coeffsList):
	LieGroup (torch::tensor(coeffsList, torch::kFloat))
{
}

template<typename Derived>
Derived &LieGroup<Derived>::operator = (const LieGroup<Derived> &other)
{
	derived().coeffs = other.coeffs;
	return derived();
}

template<typename Derived>
Derived &LieGroup<Derived>::operator = (const torch::detail::TensorDataContainer &coeffsList)
{
	assert (coeffsList.sizes().size() == 1 && coeffsList.sizes()[0] == Dim && "Incompatible initializer list size");

	derived().coeffs = torch::tensor(coeffsList, torch::kFloat);
	return derived();
}

template<class Derived>
Derived LieGroup<Derived>::inverse() const {
	return derived().inverse ();
}

template<class Derived>
typename LieGroup<Derived>::Tangent
LieGroup<Derived>::log() const {
	return derived().log ();
}

template<class Derived>
Derived LieGroup<Derived>::compose(const Derived &other) const {
	return derived().compose (other);
}

template<class Derived>
typename LieGroup<Derived>::Vector
LieGroup<Derived>::act(const Vector &v) const {
	return derived().act (v);
}

template<class Derived>
typename LieGroup<Derived>::Tangent
LieGroup<Derived>::differentiate (const Vector &outerGradient, const Vector &v) const {
	return derived().differentiate (outerGradient, v);
}

/*
template<typename Derived>
typename LieGroup<Derived>::DataType &
LieGroup<Derived>::coeffs () {
	return derived().coeffs ();
}

template<typename Derived>
const typename LieGroup<Derived>::DataType &
LieGroup<Derived>::coeffs () const {
	return derived().coeffs ();
}*/

template<typename Derived>
Derived &LieGroup<Derived>::setIdentity()
{
	const static Tangent zero = Tangent::Zero ();
	derived() = zero.exp ();
	return derived ();
}

template<typename Derived>
Derived LieGroup<Derived>::Identity () {
	const static Tangent zero = Tangent::Zero ();
	return zero.exp ();
}



template<typename Derived>
Derived LieGroup<Derived>::plus (const Tangent &other) const {
	return compose (other.exp ());
}

template<typename Derived>
typename LieGroup<Derived>::Tangent
LieGroup<Derived>::minus (const Derived &other) const {
	return other.inverse().compose (derived()).log ();
}

template<typename Derived>
Derived LieGroup<Derived>::operator + (const LieGroup<Derived>::Tangent &t) const {
	return plus(t);
}

template<typename Derived>
typename LieGroup<Derived>::Tangent
LieGroup<Derived>::operator - (const Derived &t) const {
	return minus(t);
}

template<typename Derived>
Derived LieGroup<Derived>::operator - (const Tangent &t) const {
	return plus(-t);
}

template<typename Derived>
bool LieGroup<Derived>::isApprox (const Derived &other) const {
	return minus(other).isApprox (Tangent::Zero ());
}

template<typename Derived>
Derived LieGroup<Derived>::operator * (const Derived &other) const {
	return compose (other);
}

template<typename Derived>
typename LieGroup<Derived>::Vector
LieGroup<Derived>::operator * (const LieGroup<Derived>::Vector &v) const {
	return act(v);
}

template<class Derived>
template<class DerivedOther>
typename LieGroup<Derived>::Vector LieGroup<Derived>::operator *(const DerivedOther &other) const
{
	return act(other.coeffs);
}

}

#endif // LIEGROUP_H
