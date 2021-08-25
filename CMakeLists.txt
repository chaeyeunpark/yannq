cmake_minimum_required(VERSION 3.13)

project(Yannq)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_STANDARD 17)
set(LINK_WHAT_YOU_USE TRUE)
set(BUILD_RPATH_USE_ORIGIN TRUE)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(JSON_URL "https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp")
set(JSON_DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/externals/nlohmann")
set(JSON_DOWNLOAD_PATH "${JSON_DOWNLOAD_DIR}/json.hpp")

if (NOT EXISTS "${JSON_DOWNLOAD_PATH}")
	file(MAKE_DIRECTORY "${JSON_DOWNLOAD_DIR}")
	file(DOWNLOAD "${JSON_URL}" "${JSON_DOWNLOAD_PATH}")
endif()


if (DEFINED ENV{MKLROOT})
	message(STATUS "MKL is found at $ENV{MKLROOT}.")
else()
	message(FATAL_ERROR "MKL is not found; CMake will exit.")
endif()


add_library(yannq INTERFACE)

if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
	message(STATUS "Clang (or Intel one API) compiler detected.")
	target_link_libraries(yannq INTERFACE mkl_intel_lp64 mkl_tbb_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	message(STATUS "GNU compiler detected.")
	target_link_libraries(yannq INTERFACE mkl_intel_lp64 mkl_tbb_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	message(STATUS "Intel compiler detected.")
	target_compile_options(yannq INTERFACE "-mkl=parallel;-tbb;")
	target_link_libraries(yannq INTERFACE tbb stdc++ pthread m dl)
else()
	message(WARNING "Unknown compiler. Set your own compile scripts.")
endif()


target_include_directories(yannq INTERFACE "${PROJECT_SOURCE_DIR}/externals" "${PROJECT_SOURCE_DIR}/cereal/include/" "${PROJECT_SOURCE_DIR}/Yannq" "$ENV{MKLROOT}/include")
target_link_libraries(yannq INTERFACE Eigen3::Eigen)
target_compile_options(yannq INTERFACE "-DEIGEN_USE_MKL_ALL")

if (BUILD_TESTS)
	add_subdirectory(Tests)
endif()

if (BUILD_EXAMPLES)
	add_subdirectory(Examples)
endif()
