#include <torch/all.h>
#include <iostream>

#include "../../sparcsnode/include/sparcsnode/profiling.h"
#include "../../sparcsnode/include/sparcsnode/utils.h"



using namespace torch;
using namespace std;

#define N 10000

Tensor x (const Tensor &q) {
	return (q.dim() == 1? q.unsqueeze(0) : q).slice(1,0,1).squeeze(1);
}

Tensor y (const Tensor &q) {
	return (q.dim() == 1? q.unsqueeze(0) : q).slice(1,1,2).squeeze(1);
}

Tensor z (const Tensor &q) {
	return (q.dim() == 1? q.unsqueeze(0) : q).slice(1,2,3).squeeze(1);
}

Tensor w (const Tensor &q) {
	return (q.dim() == 1? q.unsqueeze(0) : q).slice(1,3,4).squeeze(1);
}

Tensor actionJacobianCat (const Tensor &q, const Tensor &v) {
	Tensor j0 = torch::cat ({2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q),
						2*v[0]*y(q) - 2*v[1]*x(q) - 2*v[2]*w(q),
						2*v[0]*z(q) + 2*v[1]*w(q) - 2*v[2]*x(q)});
	Tensor j1 = torch::cat ({2*v[0]*y(q) + 2*v[1]*x(q) + 2*v[2]*w(q),
						2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q),
						2*v[0]*w(q) + 2*v[1]*z(q) - 2*v[2]*y(q)});
	Tensor j2 = torch::cat ({2*v[0]*z(q) - 2*v[1]*w(q) + 2*v[2]*x(q),
						2*v[0]*w(q) - 2*v[1]*z(q) + 2*v[2]*y(q),
						2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q)});
	Tensor j3 = torch::cat ({2*v[0]*w(q) - 2*v[1]*z(q) + 2*v[2]*y(q),
						2*v[0]*z(q) + 2*v[1]*w(q) - 2*v[2]*x(q),
						2*v[0]*y(q) + 2*v[1]*x(q) + 2*v[2]*w(q)});
	return torch::stack ({j0,j1,j2,j3}).t();
}

Tensor jammja (const Tensor &q, const Tensor &v, Tensor &jacobianCopy) {
	cout << 2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q) << endl;
	jacobianCopy[0][0] = ( 2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q)).item();
	jacobianCopy[1][0] = ( 2*v[0]*y(q) - 2*v[1]*x(q) - 2*v[2]*w(q)).item();
	jacobianCopy[2][0] = ( 2*v[0]*z(q) + 2*v[1]*w(q) - 2*v[2]*x(q)).item();
	jacobianCopy[0][1] = (-2*v[0]*y(q) + 2*v[1]*x(q) + 2*v[2]*w(q)).item();
	jacobianCopy[1][1] = ( 2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q)).item();
	jacobianCopy[2][1] = (-2*v[0]*w(q) + 2*v[1]*z(q) - 2*v[2]*y(q)).item();
	jacobianCopy[0][2] = (-2*v[0]*z(q) - 2*v[1]*w(q) + 2*v[2]*x(q)).item();
	jacobianCopy[1][2] = ( 2*v[0]*w(q) - 2*v[1]*z(q) + 2*v[2]*y(q)).item();
	jacobianCopy[2][2] = ( 2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q)).item();
	jacobianCopy[0][3] = ( 2*v[0]*w(q) - 2*v[1]*z(q) + 2*v[2]*y(q)).item();
	jacobianCopy[1][3] = ( 2*v[0]*z(q) + 2*v[1]*w(q) - 2*v[2]*x(q)).item();
	jacobianCopy[2][3] = (-2*v[0]*y(q) + 2*v[1]*x(q) + 2*v[2]*w(q)).item();
	return jacobianCopy;
}

Tensor immag (const Tensor &q) {
	if (q.sizes().size() > 1)
		return q.slice(1,0,3);
	else
		return q.slice(0,0,3);
}

int main ()
{
	double taken, totalTaken1 = 0, totalTaken2 = 0;;

	//Tensor qq = torch::tensor({{11,12,13,14},{21,22,23,24}}, kFloat);
	Tensor qq = torch::tensor({11,12,13,14}, kFloat);

	COUTN(x(qq));
	COUTN(y(qq));
	//return 0;

	cout << "New tensor jamm ja" << endl;
	for (int i = 0 ; i < N; i++) {
		Tensor qRand = torch::rand ({4}, kFloat);
		Tensor vRand = torch::rand ({3}, kFloat);
		Tensor jac1, jac2;
		Tensor jacobianCopy = torch::empty({3,4}, kFloat);
		QUA;
		PROFILE(taken,[&] {
			jac1 = actionJacobianCat(qRand, vRand);
		});
		totalTaken1 += taken;

		PROFILE(taken,[&] {
			jac2 = jammja (qRand, vRand, jacobianCopy);
		});
		totalTaken2 += taken;

	}


	cout << "Cat: total avg. " << totalTaken1/N << "ms" << endl;
	cout << "jammja: total avg. " << totalTaken2/N << "ms" << endl;
}
