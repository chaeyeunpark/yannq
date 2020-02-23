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

set (CMAKE_CXX_STANDARD 14)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set (OpenMP_C_FLAGS "-fopenmp=libiomp5")
	set (OpenMP_CXX_FLAGS "-fopenmp=libiomp5")
	set (OpenMP_EXE_LINKER_FLAGS "-fopenmp=libiomp5 -liomp5")
	link_directories("/usr/lib/x86_64-linux-gnu")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	find_package(OpenMP REQUIRED)
	link_directories("${OPENMP_LIBRARIES}")
endif()

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_USE_MKL_ALL")

add_executable(TestRBMState "Tests/TestRBMState.cpp")
target_link_libraries(TestRBMState mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
add_test(
	NAME TestRBMState
	COMMAND TestRBMState
)

add_executable(TestRatio "Tests/TestRatio.cpp")
target_link_libraries(TestRatio mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
add_test(
	NAME TestRatio
	COMMAND TestRatio
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
	DEPENDS TestRBMState TestRatio TestHamTFI TestHamKitaev
	TestHamKitaevC24 TestUtility TestHamJ1J2 TestSerializeRBM TestHamXYZSto )
	#	TestConvLayer TestActivations TestFullyConnected  TestFeedForward )

add_executable(RBMTFI "Examples/RBMTFI.cpp")
target_link_libraries(RBMTFI boost_serialization boost_filesystem boost_system boost_regex)


add_executable(TestAmplitudePhase "Tests/TestAmplitudePhase.cpp")
target_link_libraries(TestAmplitudePhase mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)

add_executable(TestLogDeriv "Tests/TestLogDeriv.cpp")
target_link_libraries(TestLogDeriv mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
