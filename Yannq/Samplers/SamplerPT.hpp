#ifndef NNQS_SAMPLERS_SAMPLERPT_HPP
#define NNQS_SAMPLERS_SAMPLERPT_HPP
#include <vector>
#include <random>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace yannq
{

template<class Machine, class RandomEngine, class StateValue, class Sweeper>
class SamplerPT
{
private:
	const Machine& qs_;
	const int n_;
	int numChain_;
	std::vector<double> betas_;

	std::vector<RandomEngine> re_;
	Sweeper& sweeper_;

protected:
	std::vector<StateValue> sv_;

public:
	SamplerPT(const Machine& qs, int numChain, Sweeper& sweeper)
		: qs_(qs), n_(qs.getN()), numChain_(numChain), sweeper_(sweeper)
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
#ifdef _OPENMP
#pragma omp parallel
		{
			numThreads = omp_get_num_threads();
		}
#else
		numThreads = 1;
#endif
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
	
	void mixChains()
	{
		using std::real;
#pragma omp parallel 
		{
#ifdef _OPENMP
			int tid = omp_get_thread_num();
#else
			int tid = 0;
#endif
			std::uniform_real_distribution<> urd(0.0,1.0);
#pragma omp for
			for(int idx = 0; idx < numChain_; idx+=2)
			{
				double p = exp((betas_[idx+1]-betas_[idx])*2.0*
						sv_[idx+1].logRatioRe(sv_[idx]));
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
						sv_[idx+1].logRatioRe(sv_[idx]));
				double u = urd(re_[tid]);
				if(u < p)
				{
					std::swap(sv_[idx+1],sv_[idx]);
				}
			}
		}
	}

	void sweep()
	{
		using std::real;
#pragma omp parallel
		{
#ifdef _OPENMP
			int tid = omp_get_thread_num();
#else
			int tid = 0;
#endif
#pragma omp for 
			for(int cidx = 0; cidx < numChain_; cidx++)
			{
				sweeper_.localSweep(sv_[cidx], betas_[cidx], re_[tid]);
			}
		}
	}


	auto sampling(int n_sweeps, int n_therm)
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
		using DataT = typename std::result_of<decltype(&StateValue::data)(StateValue)>::type;

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
}
#endif//NNQS_SAMPLERS_SAMPLERPT_HPP
