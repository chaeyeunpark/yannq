#include <iostream>
#include <iomanip>
#include <chrono>

#include <random>

#include <Eigen/Dense>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>


#include "Utilities/Utility.hpp"
using namespace nnqs;

#include "Machines/RBM.hpp"
#include "SROptimizerCG.hpp"
#include "Samplers/SimpleSamplerPT.hpp"


TEST_CASE("SRExact", "[SamplingObj]")
{
	using namespace nnqs;
	using Machine = RBM<std::complex<double>, false>;

	const int numChain = 16;
	const int n = 12;

	std::random_device rd;
	std::default_random_engine re{rd()};

	Machine qs(n, 3*n);
	qs.initializeRandom(re);

	SimpleSamplerPT ss(qs, numChain);
	ss.randomizeSigma();

	auto res = ss.sampling(2000, 400);



}

