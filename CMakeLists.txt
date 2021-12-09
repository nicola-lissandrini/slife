cmake_minimum_required(VERSION 3.0.2)
project(slife)

add_compile_options(-std=c++14)
add_compile_options(-DEIGEN_STACK_ALLOCATION_LIMIT=0)
add_compile_options(-fno-inline-small-functions)


find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
	message_generation std_msgs geometry_msgs nav_msgs)

set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
set(USE_CUDA 0)

find_package (Torch REQUIRED)

find_package(PCL REQUIRED)

catkin_package (
	INCLUDE_DIRS include ${TORCH_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS}
	LIBRARIES sparcsnode
)

include_directories(
	include
	../sparcsnode/include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS} )


add_executable(slife
	src/slife_node.cpp
	src/slife_handler.cpp
	src/optimizer.cpp
	src/landscape.cpp
	src/quaternion.cpp
	src/pose.cpp)
target_link_libraries (slife ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${PCL_LIBRARIES} sparcsnode)
add_dependencies(slife ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

