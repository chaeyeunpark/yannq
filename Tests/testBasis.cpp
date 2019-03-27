#include <vector>
#include <cstdint>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
std::vector<uint32_t> generateBasis(int n, int nup)
{
	std::vector<uint32_t> basis;
	uint32_t v = (1u<<nup)-1;
	uint32_t w;
	while(v < (1u<<n))
	{
		basis.emplace_back(v);
		uint32_t t = v | (v-1);
		w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
		v = w;
	}
	return basis;
}

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
	auto b = generateBasis(10, 5);
	uint32_t s = 0;
	REQUIRE(b.size() == binomialCoeff(10,5));
	for(auto t: b)
	{
		REQUIRE(__builtin_popcount(t) == 5);
	}
	for(int i = 0; i < b.size()-1; i++)
	{
		REQUIRE(b[i+1] > b[i]);
	}
}
