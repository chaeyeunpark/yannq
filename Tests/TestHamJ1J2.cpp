#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "HamiltonianHelper.hpp"
#include "Hamiltonians/J1J2.hpp"

TEST_CASE("Test Hamiltonian J1J2 with SignRule", "[J1J2]")
{
	constexpr int N = 12;
	const double deltas[] = {-1.0, 0.0, 1.0, 2.0};
	const bool signs[] = {true, false};

	for(auto delta: deltas)
	for(auto sign: signs)
	{
		J1J2 ham(N, 1.0, delta, sign);
		
		for(uint32_t col = 0; col < (1<<N); col++)
		{
			auto colSt = getColFromStates(ham, col);
			auto colHam = ham(col);

			REQUIRE(diffMap(colSt, colHam) < 1e-6);
		}
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
TEST_CASE("Test Stoquasity of XXX with SignRule", "[XXZJ1J2][Sto]")
{
	using std::abs;
	std::random_device rd;
	std::default_random_engine re{rd()};
	constexpr int N = 12;

	J1J2 ham(N, 1.0, 0.0, true);
	
	double s = 0;
	for(uint32_t col = 0; col < (1<<N); col++)
	{
		auto colHam = ham(col);
		s += abs(sumPositive(colHam, col));
	}
	REQUIRE(s < 1e-6);
}
