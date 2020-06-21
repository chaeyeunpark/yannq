#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <random>
#include <array>

#include "Machines/layers/FullyConnected.hpp"
#include "LayerHelper.hpp"

using namespace yannq;

TEMPLATE_TEST_CASE("Test backprop of FullyConnected layer", "[layer][fully_connected][bakward]", double, std::complex<double>)
{
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using dtype = TestType;

	using Vector = Matrix<dtype, Eigen::Dynamic, 1>;
	using Matrix = Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic>;

	std::random_device rd;
	std::default_random_engine re{rd()};
	const int inSize = 20;

	for(int outSize = 1; outSize <= 512; outSize *=2)
	{
		auto fc = FullyConnected<dtype>(inSize, outSize, true);
		fc.randomizeParams(re, 0.01);

		for(int i = 0; i < 10; i++)
		{
			Vector input = VectorXd::Random(inSize);
			Vector dout = VectorXd::Random(outSize);
			Vector din(inSize);
			Vector der(fc.paramDim());

			Vector tmp;
			fc.backprop(input, tmp, dout, din, der);

			REQUIRE_THAT((din - ndiff_in(fc, input, outSize)*dout).norm()/dout.norm()/outSize, 
					WithinAbsMatcher(0.,1e-6));

			REQUIRE_THAT((der - ndiff_weight(fc, input, outSize)*dout).norm()/dout.norm()/outSize, 
					WithinAbsMatcher(0.,1e-6));
		}
	}
}

TEMPLATE_TEST_CASE("Test get/set/update params", "[layer][fully_connected][params]", double, std::complex<double>)
{
	const int inSize = 20;
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using dtype = TestType;

	using Vector = Matrix<dtype, Eigen::Dynamic, 1>;
	using Matrix = Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic>;

	std::random_device rd;
	std::default_random_engine re{rd()};

	for(int outSize = 1; outSize <= 512; outSize *=2)
	{
		auto fc1 = FullyConnected<dtype>(inSize, outSize, true);
		auto fc2 = FullyConnected<dtype>(inSize, outSize, true);
		fc1.randomizeParams(re, 0.01);
		fc2.randomizeParams(re, 0.01);

		REQUIRE(fc1 != fc2);

		auto par1 = fc1.getParams();
		fc2.setParams(par1);

		REQUIRE(fc1 == fc2);

		fc2.setParams(Vector::Zero(fc1.paramDim()));
		fc2.updateParams(par1);

		REQUIRE(fc1 == fc2);
	}
}
