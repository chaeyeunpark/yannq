#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "Machines/RBM.hpp"
#include "States/RBMState.hpp"
#include "Utilities/Utility.hpp"

using cx_double = std::complex<double>;


TEMPLATE_TEST_CASE( "Test coeffs ratio in RBM", "[RBM][types]", (yannq::RBM<double,true>), (yannq::RBM<double,false>), (yannq::RBM<cx_double,true>), (yannq::RBM<cx_double,false>) )
{
	using namespace yannq;
	using Machine = TestType;
	using Catch::Matchers::Floating::WithinRelMatcher;

	constexpr int N = 10;
	constexpr double eps = 1e-8;
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
					auto rat = std::log(psi(v ^ (1<<toFlip)))-std::log(psi(v));
					REQUIRE( std::norm(std::exp(st.logRatio(toFlip) - rat)-1.0) <  eps);
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
					auto rat = std::log(psi(v ^ (1<<toFlip) ^ (1<<toFlip2)))
						- std::log(psi(v));
					auto rat2 = st.logRatio(toFlip,toFlip2);
					REQUIRE( std::norm(std::exp(rat2 - rat)-1.0) <  eps);
				}
			}
		}
	}
}
