#include <memory>

class A
{
public:
	static constexpr int c = 0;
	std::shared_ptr<int> b;

	A() 	{
		b = std::make_shared<int> (+c);
	}

};

int main () {
	A a;

	return 0;
}
