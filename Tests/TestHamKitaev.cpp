#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "Tests/HamiltonianHelper.hpp"
#include "Hamiltonians/KitaevHex.hpp"

TEST_CASE("Test Hamiltonian TFI", "[TFI]")
{
	constexpr double h = 0.2;
	KitaevHex ham(4, 4, 1.0);
	
	for(uint32_t col = 0; col < (1<<16); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}
