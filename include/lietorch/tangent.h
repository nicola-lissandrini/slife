#ifndef TANGENT_H
#define TANGENT_H

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include <cxxabi.h>

namespace lietorch {

namespace internal {

template<class T>
struct traits;

}

#define LIETORCH_INHERIT_TANGENT_TRAITS \
	using LieAlg = typename Base::LieAlg; \
	using LieGroup = typename Base::LieGroup; \
	using DataType = typename Base::DataType; \
	using Base::Dim;

template<class Derived>
class Tangent
{
public:
	static constexpr int Dim = internal::traits<Derived>::Dim;

	using LieAlg = typename internal::traits<Derived>::LieAlg;
	using LieGroup = typename internal::traits<Derived>::LieGroup;
	using DataType = typename internal::traits<LieGroup>::DataType;

protected:

	Derived &derived () { return *static_cast<Derived*>(this); }
	const Derived &derived () const { return *static_cast<const Derived*>(this); }

public:
	DataType coeffs;

	Tangent ();
	Tangent (const DataType &coeffs);
	Tangent (const torch::detail::TensorDataContainer &coeffsList);

	LieAlg generator (int i) const;
	LieAlg hat () const;
	LieGroup exp () const;
	Derived scale (const DataType &other) const;
	DataType norm () const;

	Derived operator - () const;
	Derived &operator = (const Tangent &t);
	Derived &operator = (const torch::detail::TensorDataContainer &coeffsList);
	Derived operator * (const DataType &other) const;
	Derived operator + (const Derived &other) const;
	Derived operator - (const Derived &other) const;
	Derived &operator += (const Derived &other);
	Derived &operator -= (const Derived &other);

	Derived &setZero();
	static Derived Zero ();
};

template<class Derived>
std::ostream &operator << (std::ostream &os, const Tangent<Derived> &l) {
	os << abi::__cxa_demangle(typeid(Derived).name(), NULL,NULL,NULL) << "\n" << l.coeffs;

	return os;
}

template<class Derived>
Tangent<Derived>::Tangent ():
	coeffs(torch::zeros({Dim},torch::kFloat))
{
}

template<class Derived>
Tangent<Derived>::Tangent (const DataType &_coeffs):
	coeffs(_coeffs)
{
}

template<class Derived>
Tangent<Derived>::Tangent (const torch::detail::TensorDataContainer &coeffsList)
{
	assert (coeffsList.sizes().size() == 1 && coeffsList.sizes()[0] == Dim && "Incompatible initializer list size");

	derived().coeffs = torch::tensor(coeffsList, torch::kFloat);
}

template<class Derived>
Derived &Tangent<Derived>::operator = (const Tangent<Derived> &other) {
	coeffs = other.coeffs;
	return derived();
}

template<class Derived>
Derived &Tangent<Derived>::operator = (const torch::detail::TensorDataContainer &coeffsList)
{
	assert (coeffsList.sizes().size() == 1 && coeffsList.sizes()[0] == Dim && "Incompatible initializer list size");

	derived().coeffs = torch::tensor(coeffsList, torch::kFloat);
	return derived();
}

template<class Derived>
Derived Tangent<Derived>::scale(const DataType &other) const {
	return derived().scale (other);
}

template<class Derived>
typename Tangent<Derived>::DataType
Tangent<Derived>::norm () const {
	return coeffs.norm ();
}

template<class Derived>
Derived Tangent<Derived>::operator * (const DataType &other) const {
	return scale (other);
}

template<class Derived>
Derived Tangent<Derived>::operator - () const {
	return Tangent (-coeffs).derived();
}

template<class Derived>
Derived Tangent<Derived>::operator + (const Derived &other) const {
	return Tangent(coeffs + other.coeffs).derived();
}

template<class Derived>
Derived Tangent<Derived>::operator - (const Derived &other) const {
	return Tangent(coeffs - other.coeffs).derived();
}

template<class Derived>
Derived &Tangent<Derived>::operator += (const Derived &other)  {
	coeffs += other.coeffs;
	return derived();
}

template<class Derived>
Derived &Tangent<Derived>::operator -= (const Derived &other)  {
	coeffs -= other.coeffs;
	return derived();
}

template<class Derived>
typename Tangent<Derived>::LieGroup
Tangent<Derived>::exp () const {
	return derived().exp ();
}

template<class Derived>
typename Tangent<Derived>::LieAlg
Tangent<Derived>::hat() const {
	return derived().hat ();
}

template<class Derived>
typename Tangent<Derived>::LieAlg
Tangent<Derived>::generator(int i) const {
	return derived().generator (i);
}

template<class Derived>
Derived &Tangent<Derived>::setZero () {
	coeffs *= 0;
	return derived();
}

template<class Derived>
Derived Tangent<Derived>::Zero()
{
	static const Tangent t(torch::zeros({Dim}, torch::kFloat));
	return t.derived();
}

}
#endif // TANGENT_H
