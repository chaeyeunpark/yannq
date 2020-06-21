#include <vector>
#include <cstdint>
#include "Basis/BasisJz.hpp"
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

uint32_t binomialCoeff(uint32_t n, uint32_t k)
{
    uint32_t res = 1;

    // Since C(n, k) = C(n, n-k)
    if ( k > n - k )
        k = n - k;

    // Calculate value of [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
    for (int i = 0; i < k; ++i)
    {
        res *= (n - i);
        res /= (i + 1);
    }

    return res;
}

TEST_CASE("GenerateBasis", "[GenBas]")
{
	std::vector<uint32_t> basis;
	for(auto t: yannq::BasisJz(10,3))
	{
		basis.emplace_back(t);
	}
	uint32_t s = 0;
	REQUIRE(basis.size() == binomialCoeff(10,3));
	for(auto t: basis)
	{
		REQUIRE(__builtin_popcount(t) == 3);
	}
	for(int i = 0; i < basis.size()-1; i++)
	{
		REQUIRE(basis[i+1] > basis[i]);
	}
}
