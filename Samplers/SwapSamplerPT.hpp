#ifndef NNQS_SAMPLERS_SWAPSAMPLERPT_HPP
#define NNQS_SAMPLERS_SWAPSAMPLERPT_HPP
#include <cmath>
#include <Eigen/Eigen>
#include <memory>
#include <omp.h>

#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

namespace nnqs
{

template<class Machine, class RandomEngine>
class SwapSamplerPT
{
public:
	using StateValueT = typename MachineStateTypes<Machine>::StateValue;

private:
	int n_;
	int numChain_;
	Machine& qs_;
	std::vector<StateValueT> sv_;
	std::vector<double> betas_;

	std::vector<RandomEngine> re_;

public:

	SwapSamplerPT(Machine& qs, int numChain)
		: n_(qs.getN()), numChain_(numChain), qs_(qs)
	{
		for(int idx = 0; idx < numChain_; idx++)
		{
			betas_.emplace_back( double(numChain_ - idx)/numChain_);
		}
	}

	void initializeRandomEngine()
	{
		std::random_device rd;
		int numThreads;
#pragma omp parallel
		{
			numThreads = omp_get_num_threads();
		}
		re_.resize(numThreads);
		for(int i = 0; i < numThreads; i++)
		{
			re_[i].seed(rd());
		}
	}
	
	void randomizeSigma()
	{
		sv_.clear();
		for(int i = 0; i < numChain_; i++)
		{
			sv_.emplace_back(qs_, randomSigma(n_, re_[0]));
		}
	}
	

	void randomizeSigma(int nup)
	{
		sv_.clear();
		for(int i = 0; i < numChain_; i++)
		{
			sv_.emplace_back(qs_, randomSigma(n_, nup, re_[0]));
		}
	}
	

	void sweep()
	{
		using std::real;
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			std::uniform_real_distribution<> urd(0.0,1.0);
			std::uniform_int_distribution<> uid(0,n_-1);
#pragma omp for 
			for(int cidx = 0; cidx < numChain_; cidx++)
			{
				for(int sidx = 0; sidx < n_; sidx++)
				{
					int swap1 = uid(re_[tid]);
					int swap2 = uid(re_[tid]);

					if(sv_[cidx].sigmaAt(swap1) == sv_[cidx].sigmaAt(swap2))
					{
						continue; //do nothing
					}

					double p = std::min(1.0,exp(betas_[cidx]*2.0*real(sv_[cidx].logRatio(swap1, swap2))));
					double u = urd(re_[tid]);
					if(u < p)//accept
					{
						sv_[cidx].flip(swap1,swap2);
					}
				}
			}
		}
	}
	void mixChains()
	{
		using std::real;
#pragma omp parallel 
		{
			int tid = omp_get_thread_num();
			std::uniform_real_distribution<> urd(0.0,1.0);
#pragma omp for 
			for(int idx = 0; idx < numChain_; idx+=2)
			{
				double p = exp((betas_[idx+1]-betas_[idx])*2.0*
						real(sv_[idx+1].logRatio(sv_[idx])));
				double u = urd(re_[tid]);
				if(u < p)
				{
					std::swap(sv_[idx+1],sv_[idx]);
				}
			}
#pragma omp for 
			for(int idx = 1; idx < numChain_-1; idx+=2)
			{
				double p = exp((betas_[idx+1]-betas_[idx])*2.0*
						real(sv_[idx+1].logRatio(sv_[idx])));
				double u = urd(re_[tid]);
				if(u < p)
				{
					std::swap(sv_[idx+1],sv_[idx]);
				}
			}
		}
	}

	auto sampling(int n_sweeps, int n_therm)
		-> std::vector<typename std::result_of<decltype(&StateValueT::data)(StateValueT)>::type>
	{

		using std::norm;
		using std::pow;
		using std::abs;
		using std::exp;

		//Thermalizing phase
		for(int n = 0; n < n_therm; n++)
		{
			sweep();
			mixChains();
		}
		using DataT = typename std::result_of<decltype(&StateValueT::data)(StateValueT)>::type;

		std::vector<DataT> res;
		res.reserve(n_sweeps);
		for(int ll = 0; ll < n_sweeps; ll++)
		{
			sweep();
			mixChains();
			res.push_back(sv_[0].data());
		}
		return res;
	}
};
} //NNQS
#endif//NNQS_SAMPLERS_SWAPSAMPLERPT_HPP
