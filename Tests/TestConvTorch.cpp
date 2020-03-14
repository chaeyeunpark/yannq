#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <random>
#include <array>
#include <torch/torch.h>

#include "Machines/torch_layers/Conv1D.cpp"

torch::Tensor translate(const torch::Tensor& x)
{
	return torch::cat({x.slice(-1,-1), x.slice(-1, 0, -1)}, -1);
}
TEST_CASE("Test translation function", "[layer][conv1d][forward]")
{
	for(int n = 0; n < 1000; n++)
	{
		torch::Tensor foo = torch::rand({4, 20});
		auto fooT = translate(foo);
		std::cout << "foo:" << foo << std::endl;
		std::cout << "fooT:" << fooT << std::endl;

		auto foo_a = foo.accessor<float,2>();
		auto fooT_a = fooT.accessor<float,2>();

		for(int i = 0; i < 4; i++)
		{
			for(int j = 0; j < 20; j++)
			{
				REQUIRE(fooT_a[i][(j+1)%20] == foo_a[i][j]);
			}
		}
	}
}

TEST_CASE("Test translational invariance of forward", "[layer][conv1d][forward]")
{
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
				auto conv = Conv1D(true, inChannels, outChannels, kernelSize);

				for(int i = 0; i < 10; i++)
				{
					auto input1 = torch::randn({1,inChannels,inSize});
					auto input2 = translate(input1);

					torch::Tensor output1 = conv.forward(input1);
					torch::Tensor output2 = conv.forward(input2);

					double norm = (translate(output1) - output2).norm().item<double>();
					REQUIRE( norm < 1e-6);
				}
			}
		}
	}
}
