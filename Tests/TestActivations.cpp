#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <Utilities/type_traits.hpp>
#include <Machines/layers/activations.hpp>
#include <Machines/layers/ActivationLayer.hpp>

#include <type_traits>
#include "LayerHelper.hpp"

using namespace yannq;
using namespace Eigen;

TEMPLATE_PRODUCT_TEST_CASE("Test ActivationLayer by activations", "[layer][activations]", (activation::Identity, activation::LnCosh, activation::Tanh, activation::ReLU, activation::LeakyReLU),(double))
{
	using std::abs;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using T = double;
	using VectorType = typename AbstractLayer<T>::VectorType;
	using MatrixType = typename AbstractLayer<T>::MatrixType;
	using VectorRefType = typename AbstractLayer<T>::VectorRefType;
	using VectorConstRefType = typename AbstractLayer<T>::VectorConstRefType;

	auto inSize = 20u;
	auto layer = ActivationLayer<T, TestType>();

	for(int i = 0; i < 100; i++)
	{
		VectorType input = VectorType::Random(inSize);
		VectorType output(inSize);
		VectorType dout = VectorType::Random(inSize);
		VectorType din(inSize);

		auto t = VectorType(10);
		layer.forward(input,output);
		layer.backprop(input, output, dout, din, t);

		T diff = input.adjoint()*din;
		diff -= T(input.adjoint()*ndiff_in(layer, input, inSize)*dout);
		REQUIRE_THAT(abs(diff)/inSize,
				WithinAbsMatcher(0.,1e-6));
	}
}
TEMPLATE_PRODUCT_TEST_CASE("Test ActivationLayer by activations", "[layer][activations]", (activation::Identity, activation::LnCosh, activation::Tanh, activation::ReLU),(std::complex<double>))
{
	using std::abs;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using T = std::complex<double>;
	using VectorType = typename AbstractLayer<T>::VectorType;
	using MatrixType = typename AbstractLayer<T>::MatrixType;
	using VectorRefType = typename AbstractLayer<T>::VectorRefType;
	using VectorConstRefType = typename AbstractLayer<T>::VectorConstRefType;

	auto inSize = 20u;
	auto layer = ActivationLayer<T, TestType>();

	for(int i = 0; i < 100; i++)
	{
		VectorType input = VectorType::Random(inSize);
		VectorType output(inSize);
		VectorType dout = VectorType::Random(inSize);
		VectorType din(inSize);

		auto t = VectorType(10);
		layer.forward(input,output);
		layer.backprop(input, output, dout, din, t);

		T diff = input.adjoint()*din;
		diff -= (input.adjoint()*ndiff_in(layer, input, inSize)*dout).value();
		REQUIRE_THAT(abs(diff)/inSize,
				WithinAbsMatcher(0.,1e-6));
	}
}
