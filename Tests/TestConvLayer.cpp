#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <random>
#include <array>

#include "Machines/layers/Conv1D.hpp"
#include "LayerHelper.hpp"

using namespace yannq;


TEST_CASE("Test backprop of Conv1D layer", "[layer][conv1d][backprop]")
{
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	std::random_device rd;
	std::default_random_engine re{rd()};

	const int inSize = 20;

	const int channels[] = {1,2,4,8,12};

	for(const int kernelSize: {3,5,7})
	{
		for(const int inChannels: channels)
		{
			for(const int outChannels: channels)
			{
				const int outSize = outChannels*inSize;
				auto conv = yannq::Conv1D<double>(inChannels, outChannels, kernelSize);
				conv.randomizeParams(re, 0.01);

				for(int i = 0; i < 10; i++)
				{
					VectorXd input = VectorXd::Random(inSize*inChannels);
					VectorXd dout = VectorXd::Random(outSize);
					VectorXd din(inSize*inChannels);
					VectorXd der(conv.paramDim());

					conv.backprop(input, VectorXd(), dout, din, der);

					REQUIRE_THAT((din - ndiff_in(conv, input, outSize)*dout).norm()/dout.norm()/outSize, 
							WithinAbsMatcher(0.,1e-6));

					REQUIRE_THAT((der - ndiff_weight(conv, input, outSize)*dout).norm()/dout.norm()/outSize, 
							WithinAbsMatcher(0.,1e-6));
				}
			}
		}
	}
}

Eigen::VectorXd translate(const Eigen::VectorXd& t, int size, int channels)
{
	Eigen::VectorXd res(t.size());
	for(int c = 0; c < channels; c++)
	{
		for(int i = 0; i < size; i++)
		{
			res(i+size*c) = t((i+1)%size + size*c);
		}
	}
	return res;
}

TEST_CASE("Test translational invariance of forward", "[layer][conv1d][forward]")
{
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	std::random_device rd;
	std::default_random_engine re{rd()};

	const int inSize = 20;

	const int channels[] = {1,2,4,8,12};

	for(const int kernelSize: {3,5,7})
	{
		for(const int inChannels: channels)
		{
			for(const int outChannels: channels)
			{
				const int outSize = outChannels*inSize;
				auto conv = yannq::Conv1D<double>(inChannels, outChannels, kernelSize);
				conv.randomizeParams(re, 0.01);

				for(int i = 0; i < 10; i++)
				{
					VectorXd input = VectorXd::Random(inSize*inChannels);
					VectorXd output1(outSize);
					VectorXd output2(outSize);

					conv.forward(input, output1);
					conv.forward(translate(input, inSize, inChannels), output2);

					REQUIRE_THAT( (translate(output1, inSize, outChannels) - output2).norm(), WithinAbsMatcher(0.,1e-6));
				}
			}
		}
	}
}

TEST_CASE("Test get/set/update params", "[layer][conv1d][params]")
{
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	std::random_device rd;
	std::default_random_engine re{rd()};

	const int inSize = 20;

	const int channels[] = {1,2,4,8,12};

	for(const int kernelSize: {3,5,7})
	{
		for(const int inChannels: channels)
		{
			for(const int outChannels: channels)
			{
				const int outSize = outChannels*inSize;
				auto conv1 = yannq::Conv1D<double>(inChannels, outChannels, kernelSize);
				auto conv2 = yannq::Conv1D<double>(inChannels, outChannels, kernelSize);
				conv1.randomizeParams(re, 0.01);
				conv2.randomizeParams(re, 0.01);

				REQUIRE(conv1 != conv2);

				auto par1 = conv1.getParams();
				conv2.setParams(par1);

				REQUIRE(conv1 == conv2);

				conv2.setParams(Eigen::VectorXd::Zero(conv1.paramDim()));
				conv2.updateParams(par1);

				REQUIRE(conv1 == conv2);
			}
		}
	}
}
