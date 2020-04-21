#ifndef NNQS_SAMPLERS_SWAPSWEEPER_HPP
#define NNQS_SAMPLERS_SWAPSWEEPER_HPP
#include <random>

namespace yannq
{
class SwapSweeper
{
private:
	const int n_;
	const int nSweep_;

public:
	SwapSweeper(int n, int nSweep = 1) noexcept
		: n_(n), nSweep_(nSweep)
	{
	}

	template<class StateValue, class RandomEngine>
	void localSweep(StateValue& sv, typename StateValue::RealScalar beta,
			RandomEngine& re) noexcept
	{
		using RealScalar = typename StateValue::RealScalar;

		std::uniform_real_distribution<RealScalar> urd(0.0,1.0);
		std::uniform_int_distribution<int> uid(0,n_-1);
		int toSweep = n_*nSweep_;
		for(int sidx = 0; sidx < toSweep; sidx++)
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
			}
		}
	}
};
} //NNQS
#endif//NNQS_SAMPLERS_SWAPSWEEPER_HPP
