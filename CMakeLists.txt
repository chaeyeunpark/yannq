cmake_minimum_required(VERSION 3.9)

project(Yannq)

include_directories(${CMAKE_SOURCE_DIR}/Yannq)
include_directories(${CMAKE_SOURCE_DIR}/externals "${CMAKE_SOURCE_DIR}/externals/cereal-v1.3.0/include/")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(LINK_WHAT_YOU_USE TRUE)
set(BUILD_RPATH_USE_ORIGIN TRUE)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

link_libraries(Eigen3::Eigen)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DEIGEN_USE_MKL_ALL")

set(CEREAL_URL "https://github.com/USCiLab/cereal/archive/v1.3.0.tar.gz")
set(CEREAL_DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/externals/cereal-v1.3.0.tar.gz")
set(CEREAL_IS_EXRACTED "${CMAKE_SOURCE_DIR}/externals/cereal_extracted")

set(JSON_URL "https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp")
set(JSON_DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/externals/nlohmann")
set(JSON_DOWNLOAD_PATH "${JSON_DOWNLOAD_DIR}/json.hpp")

set(CATCH_URL "https://github.com/catchorg/Catch2/releases/download/v2.11.1/catch.hpp")
set(CATCH_DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/externals/catch.hpp")

if (NOT EXISTS "${CEREAL_DOWNLOAD_PATH}")
	file(DOWNLOAD "${CEREAL_URL}" "${CEREAL_DOWNLOAD_PATH}")
endif()

if (NOT EXISTS "${JSON_DOWNLOAD_PATH}")
	file(MAKE_DIRECTORY "${JSON_DOWNLOAD_DIR}")
	file(DOWNLOAD "${JSON_URL}" "${JSON_DOWNLOAD_PATH}")
endif()

if (NOT EXISTS "${CATCH_URL}")
	file(DOWNLOAD "${CATCH_URL}" "${CATCH_DOWNLOAD_PATH}")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(mkl_intel_lp64 mkl_intel_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(mkl_intel_lp64 mkl_intel_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel -tbb")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(mkl_intel_lp64 mkl_tbb_thread mkl_core tbb stdc++ pthread m dl)
endif()

add_custom_command(
	OUTPUT ${CEREAL_IS_EXRACTED}
	COMMAND tar -xvf ${CEREAL_DOWNLOAD_PATH} -C ${CMAKE_SOURCE_DIR}/externals
	COMMAND touch ${CEREAL_IS_EXRACTED}
	DEPENDS ${CEREAL_DOWNLOAD_PATH}
	COMMENT "Unpacking cereal"
	VERBATIM
)

add_custom_target(libcereal_untar DEPENDS ${CEREAL_IS_EXTRACTED})

add_executable(TestRBMStates "Tests/TestRBMStates.cpp")
add_test(
	NAME TestRBMStates
	COMMAND TestRBMStates
)

add_executable(TestSerializeRBM "Tests/TestSerializeRBM.cpp")
add_dependencies(TestSerializeRBM libcereal_untar)
add_test(
	NAME TestSerializeRBM
	COMMAND TestSerializeRBM
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
add_dependencies(RBMTFI libcereal_untar)
target_link_libraries(RBMTFI stdc++fs)

add_executable(RBMXXZ "Examples/RBMXXZ.cpp")
add_dependencies(RBMXXZ libcereal_untar)
target_link_libraries(RBMXXZ stdc++fs)

add_executable(RBMTFIExact "Examples/RBMTFIExact.cpp")
add_dependencies(RBMTFIExact libcereal_untar)
target_link_libraries(RBMTFIExact stdc++fs)

add_executable(port_saved "utils/port_saved.cpp")
add_dependencies(port_saved libcereal_untar)
target_link_libraries(port_saved stdc++fs)

add_executable(TestAmplitudePhase "Tests/TestAmplitudePhase.cpp")

