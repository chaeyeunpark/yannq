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
	const int n_;
	const int nSweep_;

public:

	LocalSweeper(int n, int nSweep = 1) noexcept
		: n_(n), nSweep_(nSweep)
	{
	}

	template<class StateValue, class RandomEngine>
	void localSweep(StateValue& sv, double beta, RandomEngine& re) noexcept
	{
		std::uniform_real_distribution<double> urd(0.0, 1.0);
		std::uniform_int_distribution<int> uid_(0,n_-1);
		for(int sidx = 0; sidx < n_*nSweep_; sidx++)
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
