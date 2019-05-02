#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "Machines/RBM.hpp"
#include "States/RBMState.hpp"
#include "Utilities/Utility.hpp"
TEST_CASE("Test coeffs of RBM", "[RBM]")
{
	using namespace nnqs;
	const int N = 10;
	using Machine = RBM<std::complex<double>>;
	Machine rbm(N,N);
	std::random_device rd;
	std::default_random_engine re{0};
	std::uniform_int_distribution<int> uid{0,N-1};

	SECTION("Test single ratio")
	{
		for(int i = 0; i < 100; i++)
		{
			rbm.initializeRandom(re, 1.0);
			auto psi = getPsi(rbm, true);

			for(int k = 0; k < 10; k++)
			{
				Eigen::VectorXi sigma = randomSigma(N, re);
				uint32_t v = toValue(sigma);
				RBMStateValue<Machine> st(rbm, sigma);

				for(int m = 0; m < 5; m++)
				{
					int toFlip = uid(re);
					auto rat = psi(v ^ (1<<toFlip))/psi(v);
					REQUIRE( std::norm(st.ratio(toFlip) - rat) < 1e-6);
				}
			}
		}
	}
	SECTION("Test double ratio")
	{
		for(int i = 0; i < 100; i++)
		{
			rbm.initializeRandom(re, 1.0);
			auto psi = getPsi(rbm, true);

			for(int k = 0; k < 30; k++)
			{
				Eigen::VectorXi sigma = randomSigma(N, re);
				uint32_t v = toValue(sigma);
				RBMStateValue<Machine> st(rbm, sigma);

				for(int m = 0; m < 10; m++)
				{
					int toFlip = uid(re);
					int toFlip2 = uid(re);
					if(toFlip == toFlip2)
						continue;
					auto rat = psi(v ^ (1<<toFlip))/psi(v);
					REQUIRE( std::norm(st.ratio(toFlip) - rat) < 1e-6);
				}
			}
		}
	}
}
