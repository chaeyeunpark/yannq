#include <iostream>
#include <iomanip>
#include <chrono>
#include "NNQS.hpp"
#include "Utility.hpp"


#define CATCH_CONFIG_MAIN
#include <catch.hpp>
using namespace nnqs;


TEST_CASE("logCoh(x) should be same to log(cosh(x))", "LOGCOSH")
{
	using namespace nnqs;

	std::random_device rd;
	std::default_random_engine re{rd()};
	std::uniform_real_distribution<double> urd{-10.0,10.0};

	std::cout << std::setprecision(8);
	
	using ValT = std::complex<double>;
	using std::log;
	using std::cosh;
	using std::abs;
	for(int i = 0; i < 100000; i++)
	{
		std::complex<double> p{urd(re), urd(re)};
		REQUIRE(abs(logCosh(p)-log(cosh(p))) < 1e-7);
	}
}
