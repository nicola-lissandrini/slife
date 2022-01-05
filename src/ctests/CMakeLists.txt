cmake_minimum_required(VERSION 3.0.2)
project(ctests)

add_compile_options(-std=c++14)

set(Torch_DIR "/opt/libtorch/share/cmake/Torch")

find_package (Torch REQUIRED)
add_compile_options(-DEIGEN_STACK_ALLOCATION_LIMIT=0)
add_compile_options(-msse2)
set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)


include_directories(
	include
	../../include
	${TORCH_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	/opt/ros/noetic/include
)

add_executable(test_lietorch test_bind.cpp)
target_link_libraries (test_lietorch ${TORCH_LIBRARIES} ${PCL_LIBRARIES})
