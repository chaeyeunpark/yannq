#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "Machines/RBM.hpp"
#include "HessianHelper/RBMHessian.hpp"

TEST_CASE( "Hessians are computed", "[hessian]" )
{
	using namespace nnqs;

	const int n = 16;
	const int m = 16;
	std::random_device rd;
	std::default_random_engine re{rd()};

	RBM<double, true> qs(n, m);
	qs.initializeRandom(re);


}
