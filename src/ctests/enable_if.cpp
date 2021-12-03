#include <type_traits>
#include <functional>
#include <cassert>
#include <iostream>

using namespace  std;

template<typename bubu>
enable_if<is_same<bubu, int>::value, int>::type ciao () {
	return 145;
}
template<typename bubu>
enable_if<is_same<bubu, float>::value, int>::type ciao () {
	return 0;
}

template<class T>
class A
{
	T var;

public:
	A(T _init):
		var(_init)
	{}

	using Fcn = function<T(T)>;

	Fcn getTestFcn ();

	void test () {
		function<T(T)> testFcn;

		testFcn = getTestFcn ();

		cout << testFcn(var) << endl;
	}


};

template<typename T>
A<T>::Fcn A<T>::getTestFcn () {
	assert (false && "Cant test this type");
}

template<>
A<int>::Fcn A<int>::getTestFcn () {
	return [] (int i) {return 2  * i; };
}

template<>
A<float>::Fcn A<float>::getTestFcn () {
	return [] (float i) {return 3.14  * i; };
}
