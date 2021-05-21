#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <random>
#include <array>

#include "Machines/layers/Mean.hpp"
#include "LayerHelper.hpp"

using namespace yannq;

TEST_CASE("Test Mean layer", "[layer][mean]")
{
	using namespace Eigen;

	VectorXd in = VectorXd::LinSpaced(15, 0, 14);
	MatrixXd m = Map<const MatrixXd>(in.data(), 3, 5);
	m.transposeInPlace();
	in = Map<const VectorXd>(m.data(), 15);
	
	auto m1 = Mean<double>({5, 3}, 0);
	REQUIRE(m1.inputDim() == 15);
	REQUIRE(m1.outputDim() == 3);

	VectorXd out = VectorXd::Zero(m1.outputDim());
	m1.forward(in, out);

	REQUIRE((out - VectorXd::LinSpaced(3, 6, 8)).norm() < 1e-6);


	auto m2 = Mean<double>({5, 3}, 1);
	REQUIRE(m2.inputDim() == 15);
	REQUIRE(m2.outputDim() == 5);

	out = VectorXd::Zero(m2.outputDim());
	m2.forward(in, out);
	REQUIRE((out - VectorXd::LinSpaced(5, 1, 13)).norm() < 1e-6);
}


TEST_CASE("Test backprop of Mean layer", "[layer][mean][backprop]")
{
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	{
		auto m1 = Mean<double>({5, 3}, 0);
		int inSize = 15;
		int outSize = 3;
		for(int i = 0; i < 10; i++)
		{
			VectorXd input = VectorXd::Random(inSize);
			VectorXd dout = VectorXd::Random(outSize);
			VectorXd din(inSize);
			VectorXd der(m1.paramDim());

			m1.backprop(input, VectorXd(), dout, din, der);

			REQUIRE_THAT((din - ndiff_in(m1, input, outSize)*dout).norm()/dout.norm()/outSize, 
					WithinAbsMatcher(0.,1e-6));
		}
	}
	{
		auto m2 = Mean<double>({5, 3}, 1);
		int inSize = 15;
		int outSize = 5;
		for(int i = 0; i < 10; i++)
		{
			VectorXd input = VectorXd::Random(inSize);
			VectorXd dout = VectorXd::Random(outSize);
			VectorXd din(inSize);
			VectorXd der(m2.paramDim());

			m2.backprop(input, VectorXd(), dout, din, der);

			REQUIRE_THAT((din - ndiff_in(m2, input, outSize)*dout).norm()/dout.norm()/outSize, 
					WithinAbsMatcher(0.,1e-6));
		}
	}
}
