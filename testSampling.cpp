#include <iostream>
#include <iomanip>
#include <chrono>

#include <random>

#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>


#include "Utilities/Utility.hpp"
using namespace nnqs;


TEST_CASE("RandomSigma works well", "[SamplingObj]")
{
	const int n = 16;
	std::random_device rd;
	std::default_random_engine re{rd()};
	for(int i = 0; i < 1000; i++)
	{
		Eigen::VectorXi sigma = randomSigma(n, n/2, re);
		REQUIRE( sigma.sum() == 0 );
	}
}

