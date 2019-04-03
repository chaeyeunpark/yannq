#ifndef NNQS_SAMPLERS_LOCALSWEEPER_HPP
#define NNQS_SAMPLERS_LOCALSWEEPER_HPP
#include <cmath>
#include <Eigen/Eigen>
#include <random>

namespace nnqs
{

class LocalSweeper
{
private:
	int n_;

public:

	LocalSweeper(int n)
		: n_(n)
	{
	}

	template<class StateValue, class RandomEngine>
	void localSweep(StateValue& sv, double beta, RandomEngine& re)
	{
		std::uniform_real_distribution<double> urd(0.0, 1.0);
		std::uniform_int_distribution<int> uid_(0,n_-1);
		for(int sidx = 0; sidx < n_; sidx++)
		{
			int toFlip = uid_(re);
			double p = std::min(1.0,exp(beta*2.0*real(sv.logRatio(toFlip))));
			double u = urd(re);
			if(u < p)//accept
			{
				sv.flip(toFlip);
			}
		}
	}
};
} //NNQS
#endif//NNQS_SAMPLERS_LOCALSWEEPER_HPP
