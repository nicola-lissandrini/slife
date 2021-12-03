#include <iostream>

class B
{
public:
	virtual void fcn () = 0;
};

class B1 : public B
{
	int _a, _b;

public:
	B1 (int a, int b):
		_a(a), _b(b)
	{}

	void fcn () {
		std::cout << _a + _b << std::endl;
	}
};

class B2 : public B
{
	float _c;

public:
	B2 (float c):
		_c(c)
	{}

	void fcn () {
		std::cout << _c << std::endl;
	}
};

template<class _B, typename ...Args>
class A
{
	_B b;

public:
	A(Args &&...args):
		b(args ...)
	{}

	void fcn () {
		b.fcn ();
	}
};

int main () {
	A<B1, int, int> ab1(1,2);
	A<B2, float> ab2(1.3);

	ab1.fcn ();
	ab2.fcn ();

	return 0;
}
