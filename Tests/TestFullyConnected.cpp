#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <random>
#include <array>

#include "Machines/layers/FullyConnected.hpp"
#include "LayerHelper.hpp"

using namespace yannq;


TEST_CASE("Test backprop of FullyConnected layer", "[layer,fully_connected]")
{
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using dtype = std::complex<double>;

	using VectorType = Matrix<dtype, Eigen::Dynamic, 1>;
	using MatrixType = Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic>;

	std::random_device rd;
	std::default_random_engine re{rd()};
	const int inSize = 20;

	for(int outSize = 1; outSize <= 512; outSize *=2)
	{
		auto fc = FullyConnected<dtype>(inSize, outSize, true);
		fc.randomizeParams(re, 0.01);

		for(int i = 0; i < 10; i++)
		{
			VectorType input = VectorXd::Random(inSize);
			VectorType dout = VectorXd::Random(outSize);
			VectorType din(inSize);
			VectorType der(fc.paramDim());

			fc.backprop(input, VectorXd(), dout, din, der);

			REQUIRE_THAT((din - ndiff_in(fc, input, outSize)*dout).norm()/dout.norm()/outSize, 
					WithinAbsMatcher(0.,1e-6));

			REQUIRE_THAT((der - ndiff_weight(fc, input, outSize)*dout).norm()/dout.norm()/outSize, 
					WithinAbsMatcher(0.,1e-6));
		}
	}
}
