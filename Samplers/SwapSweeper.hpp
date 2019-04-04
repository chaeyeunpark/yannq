#ifndef NNQS_SAMPLERS_SWAPSWEEPER_HPP
#define NNQS_SAMPLERS_SWAPSWEEPER_HPP
#include <random>

namespace nnqs
{
class SwapSweeper
{
private:
	int n_;

public:
	SwapSweeper(int n)
		: n_(n)
	{
	}

	template<class StateValue, class RandomEngine>
	void localSweep(StateValue& sv, double beta, RandomEngine& re)
	{
		std::uniform_real_distribution<> urd(0.0,1.0);
		std::uniform_int_distribution<> uid(0,n_-1);
		for(int sidx = 0; sidx < n_; sidx++)
		{
			int swap1 = uid(re);
			int swap2 = uid(re);
			if(sv.sigmaAt(swap1) == sv.sigmaAt(swap2))
			{
				return ;
			}
			double p = std::min(1.0,exp(beta*2.0*real(sv.logRatio(swap1, swap2))));
			double u = urd(re);
			if(u < p)//accept
			{
				sv.flip(swap1,swap2);
			}
		}
	}
};
} //NNQS
#endif//NNQS_SAMPLERS_SWAPSWEEPER_HPP
