#include <torch/all.h>
#include "../../sparcsnode/include/sparcsnode/profiling.h"
#include "../../sparcsnode/include/sparcsnode/utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
using namespace torch;
using namespace std;
#define N 10000

auto gpu = torch::kCUDA;
auto cpu = torch::kCPU;
auto cudaOpts = TensorOptions().device (gpu).dtype (kFloat);
auto cpuOpts = TensorOptions().device (cpu).dtype (kFloat);

void donothing (const Tensor &ciao) {
}

void testNewTensors () {
	Tensor untensore = torch::empty ({7},  cudaOpts);
	double taken;

	COUT("prealloc&copy");
	PROFILE_N (taken, [&]{
	for (int i = 0; i < N; i++) {
		untensore[0] = i;
		untensore[1] = 2*i;
		untensore[2] = 3*i;
		untensore[3] = 4*i;
		untensore[4] = 5*i;
		untensore[5] = 6*i;
		untensore[6] = 7*i;

		Tensor mio = untensore.clone ();
		donothing (mio);
	}
	}, N);

	COUT("new tensor");
	PROFILE_N (taken, [&]{
	for (int i = 0; i < N; i++) {
		Tensor mio = torch::tensor ({i, 2*i, 3*i, 4*i, 5*i, 6*i, 7*i}, cudaOpts);
		donothing (mio);
	}

	}, N);

}

void testOps (torch::TensorOptions opts, bool bubu)
{
	at::globalContext ().setAllowTF32CuBLAS (bubu);
	at::globalContext ().setAllowTF32CuDNN (bubu);
	Tensor random = torch::rand ({10000,50000}, opts);
	Tensor prodo;
	float taken;

	PROFILE(taken, [&]{
		prodo = random.matmul (random.t());
	});

	COUTNS(prodo);
}

void display_c10_cuda_mem_stat(int32_t sleep_time) {
    printf("currentMemoryAllocated/[maxMemoryAllocated]: \t %0.1f/[%0.1f] MB\n ",
	   c10::cuda::CUDACachingAllocator::currentMemoryAllocated(0) / 1024.0 / 1024.0,
	   c10::cuda::CUDACachingAllocator::maxMemoryAllocated(0) / 1024.0 / 1024.0);
    printf("currentMemoryCached/[maxMemoryCached]: \t %0.1f/[%0.1f] MB\n",
	   c10::cuda::CUDACachingAllocator::currentMemoryCached(0) / 1024.0 / 1024.0,
	   c10::cuda::CUDACachingAllocator::maxMemoryCached(0) / 1024.0 / 1024.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000*sleep_time));
}


int main ()
{
	COUT("cuda");
	testOps (cudaOpts, false);
	while (true) {

		display_c10_cuda_mem_stat(1);
		testOps (cudaOpts, true);
	}

	COUT("cpu")
	testOps (cpuOpts, false);
}






