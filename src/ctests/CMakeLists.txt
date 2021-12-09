cmake_minimum_required(VERSION 3.0.2)
project(ctests)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-std=c++14)

set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
set(USE_CUDA 0)
find_package (Torch REQUIRED)
add_compile_options(-DEIGEN_STACK_ALLOCATION_LIMIT=0)
add_compile_options(-msse2)
set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)

find_package (PCL 1.3 REQUIRED)

include_directories(
	include
	../../include
	${TORCH_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	/opt/ros/noetic/include
)

add_executable(test_lietorch test_pinv.cpp)
target_link_libraries (test_lietorch ${TORCH_LIBRARIES} ${PCL_LIBRARIES})
