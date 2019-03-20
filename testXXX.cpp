#include "NNQS.hpp"
#include "SamplingResult.hpp"
#include "SimpleSampler.hpp"
#include "SROptimizer.hpp"
#include "XXX.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
using namespace nnqs;



TEST_CASE("XXX should behave well", "[XXX]")
{
	using std::abs;
	using namespace nnqs;

	const int N  = 16;
	const int alpha = 1;
	
	std::random_device rd;
	std::default_random_engine re{rd()};

	std::cout << std::setprecision(8);
	
	using ValT = std::complex<double>;
	NNQS<ValT> qs(N, N);
	qs.initializeRandom(re);
	auto psi = getPsi(qs);

	XXX ham(N);

	SECTION("XXX is the same for a state")
	{
		auto sigma = randomSigma(N, re);
		StateValue<ValT> s(&qs, sigma);
		ValT r1 = ham(s);

		auto col = toValue(sigma);
		ValT r2 = ham.getCol(col).transpose()*psi;
		r2 /= psi.coeff(col);
		REQUIRE( abs(r1 - r2) < 1e-6);
	}
}
