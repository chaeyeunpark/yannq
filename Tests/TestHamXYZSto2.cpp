#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "HamiltonianHelper.hpp"
#include "Hamiltonians/XYZSto2.hpp"
#include <random>

TEST_CASE("Test Hamiltonian XYZSto2", "[XYSto]")
{
	std::random_device rd;
	std::default_random_engine re{rd()};
	constexpr int N = 12;

	std::uniform_real_distribution<> ud(-1.0, 1.0);
	double a = ud(re);
	double b = ud(re);

	XYZNNNSto2 ham(N, a, b);
	
	for(uint32_t col = 0; col < (1<<N); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}

double sumPositive(const std::map<uint32_t, double>& m, uint32_t col)
{
	double res = 0.;
	for(const auto& elt: m)
	{
		if(elt.first != col && elt.second > 0)
		{
			res += elt.second;
		}
	}
	return res;
}
TEST_CASE("Test Stoquasity of XYZSto2", "[XYZSto][Sto]")
{
	using std::abs;
	std::random_device rd;
	std::default_random_engine re{rd()};
	constexpr int N = 12;

	std::uniform_real_distribution<> ud(1.0, 2.0);

	for(int n = 0; n < 100; n++)
	{
		double a = ud(re);
		double b = ud(re);

		XYZNNNSto2 ham(N, a, b);
		
		double s = 0;
		for(uint32_t col = 0; col < (1<<N); col++)
		{
			auto colHam = ham(col);
			s += abs(sumPositive(colHam, col));
		}
		REQUIRE(s < 1e-6);
	}
}
