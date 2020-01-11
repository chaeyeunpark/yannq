#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "HamiltonianHelper.hpp"
#include "Hamiltonians/KitaevHexC24.hpp"

TEST_CASE("Test Hamiltonian Kitaev with J", "[KiatevC24]")
{
	KitaevHexC24 ham(1.0);
	
	for(uint32_t col = 0; col < (1<<16); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}

TEST_CASE("Test Hamiltonian Kitaev anisotropic", "[KiatevC24]")
{
	KitaevHexC24 ham(1.0, 0.6, 1.4);
	
	for(uint32_t col = 0; col < (1<<16); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}

TEST_CASE("Test Hamiltonian Kitaev magnetic filed", "[KiatevC24]")
{
	constexpr double h = 0.2;
	KitaevHexC24 ham(1.0, h);
	
	for(uint32_t col = 0; col < (1<<16); col++)
	{
		auto colSt = getColFromStates(ham, col);
		auto colHam = ham(col);

		REQUIRE(diffMap(colSt, colHam) < 1e-6);
	}
}
