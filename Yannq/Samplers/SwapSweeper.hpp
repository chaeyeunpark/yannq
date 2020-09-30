#pragma once

#include <random>
#include <nlohmann/json.hpp>

namespace yannq
{
class SwapSweeper
{
private:
	const uint32_t n_;
	const uint32_t nSweep_;

public:
	SwapSweeper(uint32_t n, uint32_t nSweep = 1) noexcept
		: n_(n), nSweep_(nSweep)
	{
	}

	nlohmann::json desc() const
	{
		nlohmann::json res;
		res["name"] = "Swap Sweeper";
		res["num_sweeps_per"] = nSweep_;
		return res;
	}

	template<class StateValue, class RandomEngine>
	uint32_t sweep(StateValue& sv, typename StateValue::RealScalar beta,
			RandomEngine& re) noexcept
	{
		using RealScalar = typename StateValue::RealScalar;

		uint32_t acc = 0;
		std::uniform_real_distribution<RealScalar> urd(0.0,1.0);
		std::uniform_int_distribution<uint32_t> uid(0,n_-1);
		const uint32_t toSweep = n_*nSweep_;
		for(uint32_t sidx = 0; sidx < toSweep; sidx++)
		{
			int swap1 = uid(re);
			int swap2 = uid(re);
			if(sv.sigmaAt(swap1) == sv.sigmaAt(swap2))
			{
				continue ;
			}
			RealScalar p = std::min(1.0, 
					exp(beta*2.0*sv.logRatioRe(swap1, swap2)));
			RealScalar u = urd(re);
			if(u < p)//accept
			{
				sv.flip(swap1,swap2);
				++acc;
			}
		}
		return acc;
	}
};
} //NNQS
