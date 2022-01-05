#include <iostream>
#include <functional>

using namespace std;

class A {
public:
	using Fcn = std::function<void(int)>;
	using FcnExtra = std::function<void(int, int)>;

private:
	FcnExtra fcnExtra;
	Fcn fcn;

public:
	A (const FcnExtra &callback):
		fcnExtra(callback),
		fcn(bind (fcnExtra,
				placeholders::_1,
				394))
	{
	}

	void doStuff () {
		fcnExtra (1, 2);
		fcn(3);
	}
};



int main () {
	A::FcnExtra boh = [] (int a, int b) { cout << a << " bubu "  << b << endl; };
	A bubu(boh);

	bubu.doStuff ();
}
