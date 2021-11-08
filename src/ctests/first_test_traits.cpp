#ifndef CTSTS_H
#define CTSTS_H

#include <iostream>

using namespace std;

namespace internal {

template<typename T>
struct traits;

}

template<class Derived>
class LieGroup
{
public:
	using DataType = typename internal::traits<Derived>::DataType;
	static constexpr int Dim = internal::traits<Derived>::Dim;

	LieGroup () {
		ciao ();
	}

	void ciao () {
		cout << Dim << endl;
	}
};

class Quaternion;

namespace internal {


template<>
struct traits<Quaternion>
{
	using DataType = float;
	static constexpr int Dim = 43;
};

}

class Quaternion : public LieGroup<Quaternion>
{
	using Base = LieGroup<Quaternion>;
	using DataType = Base::DataType;

public:
	DataType mio () {
		return DataType ();
	}
};

int main () {
	Quaternion ciao;

	cout << ciao.mio() << endl;

	ciao.ciao();
	return 0;
}

#endif // CTSTS_H
