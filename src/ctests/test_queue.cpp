#include <queue>
#include <iostream>

using namespace std;

class A {
	deque<int> aa;
	int max;

public:
	A (int _max):
		max(_max)
	{}

	void incoda (int val) {
		cout << "incodato " << val << endl;
		aa.push_back (val);

		if (aa.size () > max) {
			auto bb = aa.front ();
			aa.pop_front ();
			cout << "decodato " << bb << endl;
		}
	}
	void dump () {
		cout << "queue size " << aa.size () << endl;
		cout << "front " << aa.front () << " back " << aa.back () << endl;
	}
};


int main () {
	A a(3);

	a.incoda (11);
	a.incoda (22);
	a.incoda (33);
	a.incoda (44);
	a.incoda (55);

	a.dump ();
}
