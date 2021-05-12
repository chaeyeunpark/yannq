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

TEMPLATE_PRODUCT_TEST_CASE("Test ActivationLayer by activations", "[layer][activations]", 
		(activation::Identity, activation::LnCosh, activation::WeakTanh,
		 activation::ReLU, activation::HardTanh, activation::SoftShrink,
		 activation::SoftSign),
		(double))
{
	using std::abs;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using T = typename TestType::Scalar;
	using Vector = typename AbstractLayer<T>::Vector;
	using Matrix = typename AbstractLayer<T>::Matrix;
	using VectorRef = typename AbstractLayer<T>::VectorRef;
	using VectorConstRef = typename AbstractLayer<T>::VectorConstRef;

	auto inSize = 20u;
	auto layer = ActivationLayer<T, TestType>();

	for(int i = 0; i < 100; i++)
	{
		Vector input = 2.0*Vector::Random(inSize);
		Vector output(inSize);
		Vector dout = Vector::Random(inSize);
		Vector din(inSize);

		auto t = Vector(10);
		layer.forward(input,output);
		layer.backprop(input, output, dout, din, t);
		
		Vector din_num = ndiff_in(layer, input, inSize)*dout;

		Vector diff = din - din_num;

		if(diff.norm()/inSize > 1e-4)
		{
			std::cout << "input: " << input.transpose() << std::endl;
			std::cout << "dout: " <<dout.transpose() << std::endl;
			std::cout << "din: " << din.transpose() << std::endl;
			std::cout << "din_num: " << din_num.transpose() << std::endl;
		}
		REQUIRE_THAT(diff.norm()/inSize,
				WithinAbsMatcher(0.,1e-4));
	}
}

TEMPLATE_PRODUCT_TEST_CASE("Test ActivationLayer by activations with a constant",
		"[layer][activations]", 
		(activation::Tanh, activation::LeakyReLU, activation::LeakyHardTanh, activation::Cos),
		(double))
{
	using std::abs;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using T = typename TestType::Scalar;
	using Vector = typename AbstractLayer<T>::Vector;
	using Matrix = typename AbstractLayer<T>::Matrix;
	using VectorRef = typename AbstractLayer<T>::VectorRef;
	using VectorConstRef = typename AbstractLayer<T>::VectorConstRef;

	auto inSize = 20u;
	auto layer = ActivationLayer<T, TestType>(3.0);

	for(int i = 0; i < 100; i++)
	{
		Vector input = 2.0*Vector::Random(inSize);
		Vector output(inSize);
		Vector dout = Vector::Random(inSize);
		Vector din(inSize);

		auto t = Vector(10);
		layer.forward(input,output);
		layer.backprop(input, output, dout, din, t);
		
		Vector din_num = ndiff_in(layer, input, inSize)*dout;

		Vector diff = din - din_num;

		if(diff.norm()/inSize > 1e-4)
		{
			std::cout << "input: " << input.transpose() << std::endl;
			std::cout << "dout: " <<dout.transpose() << std::endl;
			std::cout << "din: " << din.transpose() << std::endl;
			std::cout << "din_num: " << din_num.transpose() << std::endl;
		}
		REQUIRE_THAT(diff.norm()/inSize,
				WithinAbsMatcher(0.,1e-4));
	}
}

TEST_CASE("Test Tanh", "[layer][activations]")
{
	using std::abs;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using T = double;
	using Vector = typename AbstractLayer<T>::Vector;
	using Matrix = typename AbstractLayer<T>::Matrix;
	using VectorRef = typename AbstractLayer<T>::VectorRef;
	using VectorConstRef = typename AbstractLayer<T>::VectorConstRef;

	auto inSize = 20u;
	auto layer = ActivationLayer<T, activation::Tanh<T> >(3.0);

	for(int i = 0; i < 100; i++)
	{
		Vector input = 2.0*Vector::Random(inSize);
		Vector output(inSize);
		Vector dout = Vector::Random(inSize);
		Vector din(inSize);

		auto t = Vector(10);
		layer.forward(input,output);
		layer.backprop(input, output, dout, din, t);
		
		Vector din_num = ndiff_in(layer, input, inSize)*dout;

		Vector diff = din - din_num;

		if(diff.norm()/inSize > 1e-4)
		{
			std::cout << "input: " << input.transpose() << std::endl;
			std::cout << "dout: " <<dout.transpose() << std::endl;
			std::cout << "din: " << din.transpose() << std::endl;
			std::cout << "din_num: " << din_num.transpose() << std::endl;
		}
		REQUIRE_THAT(diff.norm()/inSize,
				WithinAbsMatcher(0.,1e-4));
	}
}

TEMPLATE_PRODUCT_TEST_CASE("Test ActivationLayer by activations", "[layer][activations]", (activation::Identity, activation::LnCosh, activation::Tanh, activation::ReLU),(std::complex<double>))
{
	using std::abs;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using T = typename TestType::Scalar;
	using Vector = typename AbstractLayer<T>::Vector;
	using Matrix = typename AbstractLayer<T>::Matrix;
	using VectorRef = typename AbstractLayer<T>::VectorRef;
	using VectorConstRef = typename AbstractLayer<T>::VectorConstRef;

	auto inSize = 20u;
	auto layer = ActivationLayer<T, TestType>();

	for(int i = 0; i < 100; i++)
	{
		Vector input = 2.0*Vector::Random(inSize);
		Vector output(inSize);
		Vector dout = Vector::Random(inSize);
		Vector din(inSize);

		auto t = Vector(10);
		layer.forward(input,output);
		layer.backprop(input, output, dout, din, t);

		Vector din_num = ndiff_in(layer, input, inSize)*dout;

		Vector diff = din - din_num;

		if(diff.norm()/inSize/din.norm() > 1e-4)
		{
			std::cout << "input: " << input.transpose() << std::endl;
			std::cout << "dout: " <<dout.transpose() << std::endl;
			std::cout << "din: " << din.transpose() << std::endl;
			std::cout << "din_num: " << din_num.transpose() << std::endl;
		}
		REQUIRE_THAT(diff.norm()/inSize/din.norm(),
				WithinAbsMatcher(0.,1e-4));
	}
}
