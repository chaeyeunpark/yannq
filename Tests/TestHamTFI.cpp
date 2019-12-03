#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "HamiltonianHelper.hpp"
#include "Hamiltonians/TFIsing.hpp"

TEST_CASE("Test Hamiltonian TFI", "[TFI]")
{
	constexpr int N = 12;
	constexpr double h = 0.2;
	TFIsing ham(N, 1.0, h);
	
	for(uint32_t col = 0; col < (1<<N); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}
