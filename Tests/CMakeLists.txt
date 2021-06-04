set(CATCH_URL "https://github.com/catchorg/Catch2/releases/download/v2.11.1/catch.hpp")
set(CATCH_DOWNLOAD_PATH "${PROJECT_SOURCE_DIR}/externals/catch.hpp")

if (NOT EXISTS "${CATCH_DOWNLOAD_PATH}")
	file(DOWNLOAD "${CATCH_URL}" "${CATCH_DOWNLOAD_PATH}")
endif()

enable_testing()

include_directories("${PROJECT_SOURCE_DIR}/Yannq" "${PROJECT_SOURCE_DIR}/externals" "${PROJECT_SOURCE_DIR}/cereal/include")

#RBM Machine
add_executable(TestRBM "TestRBM.cpp")
add_test(
	NAME TestRBM
	COMMAND TestRBM
)

add_executable(TestRBMStates "TestRBMStates.cpp")
add_test(
	NAME TestRBMStates
	COMMAND TestRBMStates
)

#FeedForward Machine
add_executable(TestMeanLayer "TestMeanLayer.cpp")
add_test(
	NAME TestMeanLayer
	COMMAND TestMeanLayer
)

add_executable(TestConvLayer "TestConvLayer.cpp")
add_test(
	NAME TestConvLayer
	COMMAND TestConvLayer
)

add_executable(TestActivations "TestActivations.cpp")
add_test(
	NAME TestActivations
	COMMAND TestActivations
)

add_executable(TestFullyConnected "TestFullyConnected.cpp")
add_test(
	NAME TestFullyConnected
	COMMAND TestFullyConnected
)

add_executable(TestFeedForward "TestFeedForward.cpp")
add_test(
	NAME TestFeedForward
	COMMAND TestFeedForward
)


#Test Hamiltonians
add_executable(TestHamTFI "TestHamTFI.cpp")
add_test(
	NAME TestHamTFI
	COMMAND TestHamTFI
)

add_executable(TestHamJ1J2 "TestHamJ1J2.cpp")
add_test(
	NAME TestHamJ1J2
	COMMAND TestHamJ1J2
)

add_executable(TestUtility "TestUtility.cpp")
add_test(
	NAME TestUtility
	COMMAND TestUtility
)

#Serialization
add_executable(TestSerializeEigen "TestSerializeEigen.cpp")
add_test(
	NAME TestSerializeEigen
	COMMAND TestSerializeEigen
)


add_executable(TestSerializeRBM "TestSerializeRBM.cpp")
add_test(
	NAME TestSerializeRBM
	COMMAND TestSerializeRBM
)

add_executable(TestSerializeConv1D "TestSerializeConv1D.cpp")
add_test(
	NAME TestSerializeConv1D
	COMMAND TestSerializeConv1D
)

add_executable(TestSerializeActivations "TestSerializeActivations.cpp")
add_test(
	NAME TestSerializeActivations
	COMMAND TestSerializeActivations
)

add_executable(TestSerializeFullyConnected "TestSerializeFullyConnected.cpp")
add_test(
	NAME TestSerializeFullyConnected
	COMMAND TestSerializeFullyConnected
)

add_executable(TestSerializeFeedForward "TestSerializeFeedForward.cpp")
add_test(
	NAME TestSerializeFeedForward
	COMMAND TestSerializeFeedForward
)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
	DEPENDS TestRBM TestRBMStates TestConvLayer TestActivations TestFullyConnected
	TestMeanLayer TestFeedForward TestHamTFI TestHamJ1J2 TestUtility 
	TestSerializeEigen TestSerializeRBM TestSerializeConv1D TestSerializeActivations 
	TestSerializeFullyConnected TestSerializeFeedForward
)


