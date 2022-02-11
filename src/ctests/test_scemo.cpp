#include <iostream>
#include "../../sparcsnode/include/sparcsnode/utils.h"

using namespace std;

int main ()
{
	uint ciao = 5;
	int bubu = -ciao;

	cout << TYPE(static_cast<decltype(5.)> (ciao)) << endl;

/*
	for (int i = -ciao; i < (signed)ciao; i++)
		cout << "1:" << i << endl;

	for (int i = -5; i < (signed) ciao; i++)
		cout << "2:" << i << endl;

	for (int i = - (int) ciao; i < (signed)ciao; i++)
		cout << "3:" << i << endl;*/
}
