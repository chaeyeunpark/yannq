cmake_minimum_required(VERSION 3.9)

project(Yannq)

include_directories(${CMAKE_SOURCE_DIR}/Yannq)
include_directories(${CMAKE_SOURCE_DIR}/externals)

link_directories(${CMAKE_PREFIX_PATH}/lib)
include_directories(${CMAKE_PREFIX_PATH}/include/ ${EIGEN3_INCLUDE_DIR})

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

set (CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

link_libraries(Eigen3::Eigen)
link_libraries(OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_USE_MKL_ALL -march=native")

add_executable(TestRBMStates "Tests/TestRBMStates.cpp")
target_link_libraries(TestRBMStates mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
add_test(
	NAME TestRBMStates
	COMMAND TestRBMStates
)

add_executable(TestHamTFI "Tests/TestHamTFI.cpp")
add_test(
	NAME TestHamTFI
	COMMAND TestHamTFI
)

add_executable(TestHamKitaev "Tests/TestHamKitaev.cpp")
add_test(
	NAME TestHamKitaev
	COMMAND TestHamKitaev
)

add_executable(TestHamKitaevC24 "Tests/TestHamKitaevC24.cpp")
add_test(
	NAME TestHamKitaevC24
	COMMAND TestHamKitaevC24
)


add_executable(TestUtility "Tests/TestUtility.cpp")
add_test(
	NAME TestUtility
	COMMAND TestUtility
)

add_executable(TestHamJ1J2 "Tests/TestHamJ1J2.cpp")
add_test(
	NAME TestHamJ1J2
	COMMAND TestHamJ1J2
)
add_executable(TestSerializeRBM "Tests/TestSerializeRBM.cpp")
add_test(
	NAME TestSerializeRBM
	COMMAND TestSerializeRBM
)

#[[
add_executable(TestConvLayer "Tests/TestConvLayer.cpp")
add_test(
	NAME TestConvLayer
	COMMAND TestConvLayer
)

add_executable(TestActivations "Tests/TestActivations.cpp")
add_test(
	NAME TestActivations
	COMMAND TestActivations
)

add_executable(TestFullyConnected "Tests/TestFullyConnected.cpp")
add_test(
	NAME TestFullyConnected
	COMMAND TestFullyConnected
)

add_executable(TestFeedForward "Tests/TestFeedForward.cpp")
add_test(
	NAME TestFeedForward
	COMMAND TestFeedForward
)
#]]

add_executable(TestHamXYZSto "Tests/TestHamXYZSto.cpp")
add_test(
	NAME TestHamXYZSto
	COMMAND TestHamXYZSto
)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
	DEPENDS TestRBMStates TestHamTFI TestHamKitaev
	TestHamKitaevC24 TestUtility TestHamJ1J2 TestSerializeRBM TestHamXYZSto )
	#	TestConvLayer TestActivations TestFullyConnected  TestFeedForward )

add_executable(RBMTFI "Examples/RBMTFI.cpp")
target_link_libraries(RBMTFI OpenMP::OpenMP_CXX mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl stdc++fs)

add_executable(RBMTFIExact "Examples/RBMTFIExact.cpp")
target_link_libraries(RBMTFIExact OpenMP::OpenMP_CXX mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl stdc++fs)

add_executable(TestAmplitudePhase "Tests/TestAmplitudePhase.cpp")
target_link_libraries(TestAmplitudePhase mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)

