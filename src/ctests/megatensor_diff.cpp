#include <torch/all.h>
#include "../../sparcsnode/include/sparcsnode/utils.h"

using namespace torch;

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


Tensor actionJacobian2 (const Tensor &q, const Tensor &v) {
	Tensor j0 = 2*torch::tensor({{{{ 1, 0, 0}, { 0,-1, 0}, { 0, 0,-1}},
						   {{ 0, 1, 0}, { 1, 0, 0}, { 0, 0, 0}},
						   {{ 0, 0, 1}, { 0, 0, 0}, { 1, 0, 0}},
						   {{ 0, 0, 0}, { 0, 0,-1}, { 0, 1, 0}}},
						  {{{ 0, 1, 0}, { 1, 0, 0}, { 0, 0, 0}},
						   {{-1, 0, 0}, { 0, 1, 0}, { 0, 0,-1}},
						   {{ 0, 0, 0}, { 0, 0, 1}, { 0, 1, 0}},
						   {{ 0, 0, 1}, { 0, 0, 0}, {-1, 0, 0}}},
						  {{{ 0, 0, 1}, { 0, 0, 0}, { 1, 0, 0}},
						   {{ 0, 0, 0}, { 0, 0, 1}, { 0, 1, 0}},
						   {{-1, 0, 0}, { 0,-1, 0}, { 0, 0, 1}},
						   {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 0}}},
						  {{{ 0, 0, 0}, { 0, 0,-1}, { 0, 1, 0}},
						   {{ 0, 0, 1}, { 0, 0, 0}, {-1, 0, 0}},
						   {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 0}},
						   {{ 1, 0, 0}, { 0, 1, 0}, { 0, 0, 1}}}}, kFloat);
	//COUTN(v.expand({4,3}));
	return (j0 * q.unsqueeze(1).unsqueeze(1)).sum(1).matmul(v).permute({2,0,1}).transpose(1,2);
	/*return torch::stack({(2*(j0 * q.unsqueeze(1).unsqueeze(1)).sum(0)).mm(v),
					 (2*(j1 * q.unsqueeze(1).unsqueeze(1)).sum(0)).mm(v),
					 (2*(j2 * q.unsqueeze(1).unsqueeze(1)).sum(0)).mm(v),
					 (2*(j3 * q.unsqueeze(1).unsqueeze(1)).sum(0)).mm(v)}).permute({2,0,1}).transpose(1,2);
					 */
}

Tensor actionJacobian (const Tensor &q, const Tensor &v) {
	Tensor j0 = torch::cat ({ 2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q),
						 2*v[0]*y(q) - 2*v[1]*x(q) - 2*v[2]*w(q),
						 2*v[0]*z(q) + 2*v[1]*w(q) - 2*v[2]*x(q)});

	Tensor j1 = torch::cat ({-2*v[0]*y(q) + 2*v[1]*x(q) + 2*v[2]*w(q),
						2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q),
						-2*v[0]*w(q) + 2*v[1]*z(q) - 2*v[2]*y(q)});

	Tensor j2 = torch::cat ({-2*v[0]*z(q) - 2*v[1]*w(q) + 2*v[2]*x(q),
						2*v[0]*w(q) - 2*v[1]*z(q) + 2*v[2]*y(q),
						2*v[0]*x(q) + 2*v[1]*y(q) + 2*v[2]*z(q)});

	Tensor j3 = torch::cat ({ 2*v[0]*w(q) - 2*v[1]*z(q) + 2*v[2]*y(q),
						 2*v[0]*z(q) + 2*v[1]*w(q) - 2*v[2]*x(q),
						 -2*v[0]*y(q) + 2*v[1]*x(q) + 2*v[2]*w(q)});

	return torch::stack({j0,j1,j2,j3}).t();
}


int main () {
	Tensor q = torch::tensor ({1, 2, 3, 4}, kFloat);
	Tensor v = torch::rand ({3,7}, kFloat);
	Tensor outerGradient = torch::rand ({7,3});
	COUTN(v);
	COUTN(outerGradient.unsqueeze(1));
	auto jac = actionJacobian2(q,v);
	COUTN(outerGradient.unsqueeze(1).matmul(jac).squeeze())
	//COUTN(outerGradient.unsqueeze(0).mm (actionJacobian(q,v.t()[0])).squeeze());
}
