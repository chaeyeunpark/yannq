#pragma once
#include <cmath>
#include <Eigen/Eigen>
#include <random>

#include <nlohmann/json.hpp>

namespace yannq
{

class LocalSweeper
{
private:
	const int n_;
	const int nSweep_;

public:

	LocalSweeper(int n, int nSweep = 1) noexcept
		: n_(n), nSweep_(nSweep)
	{
	}

	nlohmann::json desc() const
	{
		nlohmann::json res;
		res["name"] = "Local Sweeper";
		res["num_sweep_per"] = nSweep_;
		return res;
	}

	template<class StateValue, class RandomEngine>
	void localSweep(StateValue& sv, typename StateValue::RealScalar beta, 
			RandomEngine& re) const noexcept
	{
		using RealScalar = typename StateValue::RealScalar;

		std::uniform_real_distribution<RealScalar> urd(0.0, 1.0);
		std::uniform_int_distribution<int> uid_(0,n_-1);
		for(int sidx = 0; sidx < n_*nSweep_; sidx++)
		{
			int toFlip = uid_(re);
			RealScalar p = std::min(1.0,exp(beta*2.0*sv.logRatioRe(toFlip)));
			RealScalar u = urd(re);
			if(u < p)//accept
			{
				sv.flip(toFlip);
			}
		}
	}
};
} //NNQS
