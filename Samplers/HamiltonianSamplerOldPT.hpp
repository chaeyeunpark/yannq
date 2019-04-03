#ifndef CY_HAMILTONIAN_SAMPLER_PT_HPP
#define CY_HAMILTONIAN_SAMPLER_PT_HPP
#include <cmath>
#include <Eigen/Eigen>
#include <memory>
#include <omp.h>
#include <iostream>

#include "Utilities/Utility.hpp"

namespace nnqs
{

template<class Machine, class RandomEngine, int N>
class HamiltonianSamplerPT
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
	std::vector<std::array<int, N> > flips_;

public:
	HamiltonianSamplerPT(Machine& qs, int numChain, const std::vector<std::array<int,N> >& flips)
		: n_(qs.getN()), numChain_(numChain), qs_(qs), flips_(flips)
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
	

	void sweep()
	{
		using std::real;
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			std::uniform_real_distribution<> urd(0.0,1.0);
			std::uniform_int_distribution<> uid(0,flips_.size()-1);
#pragma omp for schedule(static, 4)
			for(int cidx = 0; cidx < numChain_; cidx++)
			{
				for(int sidx = 0; sidx < n_; sidx++)
				{
					auto toFlip = flips_[uid(re_[tid])];
					double p = std::min(1.0,exp(betas_[cidx]*2.0*real(sv_[cidx].logRatio(toFlip))));
					double u = urd(re_[tid]);
					if(u < p)//accept
					{
						sv_[cidx].flip(toFlip);
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
#pragma omp for schedule(static,2)
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
#pragma omp for schedule(static,2)
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
#endif//CY_HAMILTONIAN_SAMPLER_PT_HPP
