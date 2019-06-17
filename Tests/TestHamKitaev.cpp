#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "HamiltonianHelper.hpp"
#include "Hamiltonians/KitaevHex.hpp"

TEST_CASE("Test Hamiltonian Kitaev with J", "[Kiatev]")
{
	KitaevHex ham(4, 4, 1.0);
	
	for(uint32_t col = 0; col < (1<<16); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}

TEST_CASE("Test Hamiltonian Kitaev anisotropic", "[Kiatev]")
{
	KitaevHex ham(4, 4, 0.2, 0.4, 0.6);
	
	for(uint32_t col = 0; col < (1<<16); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}

TEST_CASE("Test Hamiltonian Kitaev magnetic filed", "[Kiatev]")
{
	constexpr double h = 0.2;
	KitaevHex ham(4, 4, 1.0, h);
	
	for(uint32_t col = 0; col < (1<<16); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

			REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}
