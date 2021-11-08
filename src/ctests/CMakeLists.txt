cmake_minimum_required(VERSION 3.0.2)
project(ctests)

add_compile_options(-std=c++14)

set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
set(USE_CUDA 0)
find_package (Torch REQUIRED)


include_directories(
	include
	../../include
	${TORCH_INCLUDE_DIRS}
)

add_executable(test_lietorch test_template.cpp)
target_link_libraries (test_lietorch ${TORCH_LIBRARIES})
