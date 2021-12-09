#include "../../sparcsnode/include/sparcsnode/utils.h"
#include "../../sparcsnode/include/sparcsnode/profiling.h"
#include <torch/all.h>
//#include "../include/lietorch/quaternion.h"

using namespace std;
using namespace torch;

// Single element or vector of selected component from the tensor
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

#define N 1000
Tensor composition (const Tensor &q1, const Tensor &q2) {
	return torch::cat({w(q1)*x(q2) + x(q1)*w(q2) + y(q1)*z(q2) - z(q1)*y(q2),
				    w(q1)*y(q2) - x(q1)*z(q2) + y(q1)*w(q2) + z(q1)*x(q2),
				    w(q1)*z(q2) + x(q1)*y(q2) - y(q1)*x(q2) + z(q1)*w(q2),
				    w(q1)*w(q2) - x(q1)*x(q2) - y(q1)*y(q2) - z(q1)*z(q2)});
}


static const Tensor compositionMultiplier = torch::tensor({{{ 0, 0, 0, 1}, { 0, 0, 1, 0}, { 0,-1, 0, 0}, {-1, 0, 0, 0}},
											    {{ 0, 0,-1, 0}, { 0, 0, 0, 1}, { 1, 0, 0, 0}, { 0,-1, 0, 0}},
											    {{ 0, 1, 0, 0}, {-1, 0, 0, 0}, { 0, 0, 0, 1}, { 0, 0,-1, 0}},
											    {{ 1, 0, 0, 0}, { 0, 1, 0, 0}, { 0, 0, 1, 0}, { 0, 0, 0, 1}}}, kFloat);

static const Tensor skewMultiplier = torch::tensor({{{ 0, 0, 0}, { 0, 0,-1}, { 0, 1, 0}},
										  {{ 0, 0, 1}, { 0, 0, 0}, {-1, 0, 0}},
										  {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 0}}}, kFloat);

static const Tensor rotationMultiplier = torch::tensor({{{ 1, 0, 0, 0},
											  { 0,-1, 0, 0},
											  { 0, 0,-1, 0},
											  { 0, 0, 0, 1}},
											 {{ 0, 2, 0, 0},
											  { 0, 0, 0, 0},
											  { 0, 0, 0, 0},
											  { 0, 0,-2, 0}},
											 {{ 0, 0, 2, 0},
											  { 0, 0, 0, 0},
											  { 0, 0, 0, 0},
											  { 0, 2, 0, 0}},
											 {{ 0, 2, 0, 0},
											  { 0, 0, 0, 0},
											  { 0, 0, 0, 0},
											  { 0, 0, 2, 0}},
											 {{-1, 0, 0, 0},
											  { 0, 1, 0, 0},
											  { 0, 0,-1, 0},
											  { 0, 0, 0, 1}},
											 {{ 0, 0, 0, 0},
											  { 0, 0, 2, 0},
											  { 0, 0, 0, 0},
											  {-2, 0, 0, 0}},
											 {{ 0, 0, 2, 0},
											  { 0, 0, 0, 0},
											  { 0, 0, 0, 0},
											  { 0,-2, 0, 0}},
											 {{ 0, 0, 0, 0},
											  { 0, 0, 2, 0},
											  { 0, 0, 0, 0},
											  { 2, 0, 0, 0}},
											 {{-1, 0, 0, 0},
											  { 0,-1, 0, 0},
											  { 0, 0, 1, 0},
											  { 0, 0, 0, 1}}}, kFloat);

Tensor composition2 (const Tensor &q1, const Tensor &q2) {
	return (compositionMultiplier * q2.unsqueeze(1).unsqueeze(1)).sum(0).matmul(q1);
}

Tensor actionJacobian (const Tensor &q, const Tensor &v) {
	return q.unsqueeze(0).unsqueeze(1).matmul(rotationMultiplier)
			.matmul(q.unsqueeze(1)
				    .unsqueeze(0))
			.squeeze().reshape({3,3})
			.unsqueeze(0)
			.matmul((skewMultiplier.unsqueeze(3) *
				    v.t().unsqueeze(1).unsqueeze(1)
				    ).sum(0)
					.permute({2,0,1}));
}

Tensor qimag (const Tensor &q) {
	return q.slice(q.dim() > 1? 1 : 0, 0, 3);
}


Tensor actionJacobianSchifo (const Tensor &q, const Tensor &v) {
	Tensor jacSchifo = torch::empty({3,3}, kFloat);

	jacSchifo[0][0] = (w(q)*w(q) + x(q)*x(q) - y(q)*y(q) - z(q)*z(q)).item();
	jacSchifo[0][1] = (2*(x(q)*y(q) - w(q)*z(q))).item();
	jacSchifo[0][2] = (2*(x(q)*z(q) + w(q)*y(q))).item();
	jacSchifo[1][0] = (2*(x(q)*y(q) + w(q)*z(q))).item();
	jacSchifo[1][1] = (w(q)*w(q) - x(q)*x(q) + y(q)*y(q) - z(q)*z(q)).item();
	jacSchifo[1][2] = (2*(y(q)*z(q) - w(q)*x(q))).item();
	jacSchifo[2][0] = (2*(x(q)*z(q) - w(q)*y(q))).item();
	jacSchifo[2][1] = (2*(y(q)*z(q) + w(q)*x(q))).item();
	jacSchifo[2][2] = (w(q)*w(q) - x(q)*x(q) - y(q)*y(q) + z(q)*z(q)).item();

	return jacSchifo.matmul((skewMultiplier * v.unsqueeze(1).unsqueeze(1)).sum(0));
}


Tensor qlog (const Tensor &q) {
	Tensor imagPart = qimag(q);
	Tensor imagPartNorm = imagPart.norm();
	return 2 * imagPart / imagPartNorm * torch::acos(w(q));
}

Tensor qexp (const Tensor &t) {
	Tensor theta = t.norm(2,0);

	return torch::cat ({t / theta * (theta / 2).sin(), (theta / 2).cos().unsqueeze(0)});
}

int main () {
	Tensor a = torch::tensor({ 0.049418, 0.1492514, 0.0074688, 0.9875354}, kFloat);
	Tensor v = torch::tensor({{1,2,3},{10,20,30}}, kFloat);
	Tensor r = torch::rand({N,4}, kFloat);
	Tensor c;

	COUTN(qexp(qlog(a)) - a);

	/*
	double taken;
	PROFILE_N (taken, [&]{
		for (int i = 0; i < N; i++)

			c = composition(r[i], a);
	}, N);

	PROFILE_N (taken, [&]{
		for (int i = 0; i < N; i++)
			c = composition2 (r[i], a);
	}, N);*/
}
