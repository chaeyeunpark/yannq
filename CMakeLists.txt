cmake_minimum_required(VERSION 3.9)

project(Yannq)
include_directories(${CMAKE_SOURCE_DIR}/Yannq)

link_directories(${CMAKE_PREFIX_PATH}/lib)
include_directories(${CMAKE_PREFIX_PATH}/include/ ${EIGEN3_INCLUDE_DIR} ${CATCH2_INCLUDE_DIR})

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-int-in-bool-context")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(OpenMP)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


#set (CMAKE_CXX_FLAGS_RELEASE "-fopenmp")


add_executable(TestRBMState "Tests/TestRBMState.cpp")
add_executable(TestHamTFI "Tests/TestHamTFI.cpp")
add_executable(TestHamKitaev "Tests/TestHamKitaev.cpp")


