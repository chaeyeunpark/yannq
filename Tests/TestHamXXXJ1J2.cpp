#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "HamiltonianHelper.hpp"
#include "Hamiltonians/XXXJ1J2.hpp"

TEST_CASE("Test Hamiltonian J1J2", "[TFI]")
{
	constexpr int N = 12;
	constexpr double J2 = 0.2;
	XXXJ1J2 ham(N, 1.0, J2);
	
	for(uint32_t col = 0; col < (1<<N); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}
