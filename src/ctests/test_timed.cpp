#include <iostream>
#include <chrono>
#include <thread>
#include <queue>
#include <algorithm>
#include "../../sparcsnode/include/sparcsnode/utils.h"

template<class T, class clock = std::chrono::steady_clock>
class Timed
{
	T _obj;
	std::chrono::time_point<clock> _time;
	int bubu;

public:
	template<typename ...Args>
	Timed (const std::chrono::time_point<clock> &time, Args ...args):
		_obj(args ...),
		_time(time)
	{}

	template<typename ...Args>
	Timed (Args ...args):
		Timed (clock::now (), args ...)
	{}

	T &obj() {
		return _obj;
	}

	const T &obj () const {
		return _obj;
	}

	std::chrono::time_point<clock> &time () {
		return _time;
	}

	const std::chrono::time_point<clock> &time () const {
		return _time;
	}
};

class A
{
	int a;

public:
	A (int _a):
		a(_a)
	{}

	void dump () const {
		std::cout << a << std::endl;
	}
};

template<class T>
void foo (const Timed<T> &bubu) {
	bubu.obj ().dump ();
}

void pausee () {
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

template<typename T>
void dumpTwo (const T &it) {
	auto first = *it;
	auto before = *(it-1);

	first.obj().dump ();
	before.obj().dump ();
}

int main ()
{
	std::deque<Timed<A>> objs;

	auto primaTime = std::chrono::steady_clock::now ();
	objs.push_back (Timed<A> (1));
	pausee();
	objs.push_back (Timed<A> (2));
	pausee();
	objs.push_back (Timed<A> (3));
	pausee();
	objs.push_back (Timed<A> (4));
	pausee();
	auto pclTime = std::chrono::steady_clock::now ();

	std::chrono::duration<float> dur = pclTime - primaTime;

    COUTN((std::chrono::duration<float, std::milli> (dur)).count ());

	auto closest = std::lower_bound (objs.begin (), objs.end (), pclTime, [](const Timed<A> &gt, const decltype(pclTime) &pt) { return gt.time () < pt; });

	dumpTwo (closest);
}
