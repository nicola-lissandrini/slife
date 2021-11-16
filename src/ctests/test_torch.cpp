#include "lietorch/liegroup.h"
#include "../../sparcslib/include/profiling.h"

using namespace lietorch;
using namespace std;
using namespace torch;
using namespace torch::indexing;

#define N 400000

int main ()
{
	double taken;
	Tensor aa = torch::randn({3}, kFloat);
	cout << "float" << endl;
	float f;
	PROFILE_N (taken, [&]{
		for (int i = 0; i < N; i++) {
			f = aa.index({indexing::Slice(1,indexing::None)}).norm().item().toFloat();
		}
	}, N);

	cout << "tensor each time" << endl;
	PROFILE_N (taken, [&]{
		for (int i = 0; i < N; i++) {
			Tensor c = aa.index({indexing::Slice(1,indexing::None)}).norm();
		}
	}, N);

	cout << "prealloc tensor" << endl;
	Tensor bb = torch::empty({3}, kFloat);
	PROFILE_N (taken, [&]{
		for (int i = 0; i < N; i++) {
			bb[0] = 1;
			bb[1] = 2;
			bb[2] = 3;
		}
	}, N);
	cout << bb << endl;

	cout << "new tensor" << endl;
	PROFILE_N (taken, [&]{
		for (int i = 0; i < N; i++) {
			Tensor c = torch::tensor({1,2,3},kFloat);
		}
	}, N);

	cout << "concat tensors" << endl;
	Tensor a = torch::tensor({1}, kFloat);
	Tensor b = torch::tensor({2}, kFloat);
	Tensor c = torch::tensor({3}, kFloat);
	Tensor d;

	PROFILE_N (taken, [&]{
		for (int i = 0; i < N; i++) {
			 d = torch::cat({a,b,c});
		}
	}, N);

	cout << d << endl;

	{
		Tensor a = torch::rand ({N,3}, kFloat);
		Tensor b;
		cout << "norm w/ tensor" << endl;
		PROFILE_N (taken, [&]{
			b = a.norm(2,1);
		}, N);
		b = torch::empty ({N}, kFloat);
		cout << "norm w/ for loop" << endl;
		PROFILE_N (taken, [&]{
			for (int i = 0; i < N; i++) {
				b[i] = a[i].norm();
			}
		}, N);

		vector<int> subsLens = {1, 5, 10, 100, 1000, 20000};
		for (int subsLen : subsLens) {
			cout << "select random subset size " << subsLen << endl;
			PROFILE (taken, [&]{
				Tensor idxes = torch::randperm(N).index({Slice(0,subsLen)});
				b = NAN * a.index({idxes, Ellipsis});
				a.index ({idxes, Ellipsis}) = b;
			});
			Tensor c;
			PROFILE (taken, [&]{
				torch::Tensor validIdxes = (torch::isfinite(b).sum(1)).nonzero();
				c = b.index ({validIdxes}).view ({validIdxes.size(0), 3});
			});
		}

	}

	return 0;
}
