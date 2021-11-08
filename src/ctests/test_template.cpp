#include <iostream>
#include <sstream>
#include <math.h>

using namespace std;

template<typename T>
int ciao (T t) {
	return (int) t;
}

const auto ciaoDouble = ciao<double>;
const auto ciaoFloat = ciao<float>;

class A
{
	int a, b;

public:
	A (int _a, int _b):
		a(_a), b(_b)
	{}

	string out () {
		stringstream ss;
		ss << a << " " << b << " ";
		return ss.str();
	}
};

class B
{
	A aa[2];
public:
	B():
		aa{A(1,2),A(3,4)}
	{}

	void out () {
		cout << aa[0].out () << aa[1].out() << endl;
	}
};

int main ()
{
	B b;

	b.out();
}
