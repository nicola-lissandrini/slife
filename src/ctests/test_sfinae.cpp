#include <iostream>
#include <type_traits>


template<class T>
class A
{
public:
    template<typename Tp = T>
    typename std::enable_if<std::is_floating_point<Tp>::value, Tp>::type foo () {
        return 1.23;
    }
    template<typename Tp = T>
    typename std::enable_if<std::is_integral<Tp>::value, Tp>::type foo () {
        return 1.23;
    }
};

int main () {
    A<float> a;
    A<int> b;

    std::cout << a.foo () << std::endl;
    std::cout << b.foo () << std::endl;
}
