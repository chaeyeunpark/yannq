#ifndef NNQS_SAMPLERS_LOCALSWEEPER_HPP
#define NNQS_SAMPLERS_LOCALSWEEPER_HPP
#include <cmath>
#include <Eigen/Eigen>
#include <random>

namespace yannq
{

class LocalSweeper
{
private:
	const uint32_t n_;
	const uint32_t nSweep_;

public:

	LocalSweeper(uint32_t n, uint32_t nSweep = 1) noexcept
		: n_(n), nSweep_(nSweep)
	{
	}

	template<class StateValue, class RandomEngine>
	uint32_t sweep(StateValue& sv, typename StateValue::RealScalar beta, 
			RandomEngine& re) const noexcept
	{
		using RealScalar = typename StateValue::RealScalar;

		uint32_t acc = 0;
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
				++acc;
			}
		}
		return acc;
	}
};
} //NNQS
#endif//NNQS_SAMPLERS_LOCALSWEEPER_HPP
