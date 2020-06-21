#ifndef NNQS_SAMPLERS_SAMPLERPT_HPP
#define NNQS_SAMPLERS_SAMPLERPT_HPP
#include <vector>
#include <random>
#include <complex>

#include <tbb/tbb.h>

namespace yannq
{

template<class Machine, class RandomEngine, class StateValue, class Sweeper>
class SamplerMT
{
public:
	using Scalar = typename Machine::Scalar;
	using RealScalar = typename Machine::RealScalar;

private:
	const Machine& qs_;
	const uint32_t n_;
	const uint32_t nTmp_;
	const uint32_t nChainsPer_;

	std::vector<RealScalar> betas_; //nTmp
	std::vector<StateValue> sv_; //nTmp x nChainsPer

	std::vector<uint32_t> mixOdds;
	tbb::enumerable_thread_specific<RandomEngine> re_;
	Sweeper& sweeper_;


public:
	/**
	 * @param qs quantum state to use
	 * @param nTmp number of temperatures to use
	 * @nchainPer number of chains per each temperature
	 * @sweeper Sweeper to use
	 */
	SamplerMT(const Machine& qs, uint32_t nTmp, uint32_t nChainsPer, Sweeper& sweeper)
		: qs_{qs}, n_{qs.getN()}, nTmp_{nTmp}, nChainsPer_{nChainsPer},
		sweeper_{sweeper}
	{
		for(uint32_t idx = 0; idx < nTmp; idx++)
		{
			betas_.emplace_back( RealScalar(nTmp - idx)/nTmp );
		}
		for(uint32_t chainIdx = 0; chainIdx < nChainsPer; ++chainIdx)
		{
			for(uint32_t tmpIdx = 1; tmpIdx < nTmp-1; tmpIdx += 2)
			{
				mixOdds.emplace_back(nTmp*chainIdx+tmpIdx);
			}
		}
	}

	void initializeRandomEngine()
	{
		std::random_device rd;
		for(auto iter = re_.begin(); iter != re_.end(); ++iter)
		{
			iter->seed(rd());
		}
	}
	
	void randomizeSigma()
	{
		sv_.clear();
		for(uint32_t idx = 0u; idx < nTmp_*nChainsPer_; ++idx)
		{
			sv_.emplace_back(qs_, randomSigma(n_, re_.local()));
		}
	}

	void randomizeSigma(int nup)
	{
		sv_.clear();
		for(uint32_t idx = 0u; idx < nTmp_*nChainsPer_; ++idx)
		{
			sv_.emplace_back(qs_, randomSigma(n_, nup, re_.local()));
		}
	}

	inline double beta(uint32_t idx) const
	{
		return betas_[idx % nTmp_];
	}

	/**
	 * Mix chainss with different temperatures.
	 */
	void mixChains()
	{
		using std::real;
		tbb::enumerable_thread_specific<
			std::uniform_real_distribution<RealScalar> > urd(0.0,1.0);

		tbb::parallel_for(0u, nTmp_*nChainsPer_, 2u,
			[&](uint32_t idx)
		{
			RealScalar p = exp((beta(idx+1)-beta(idx))*2.0*
					sv_[idx+1].logRatioRe(sv_[idx]));
			RealScalar u = urd.local()(re_.local());
			if(u < p)
			{
				std::swap(sv_[idx+1],sv_[idx]);
			}
		});
		tbb::parallel_for(0u, mixOdds.size(), 
			[&](uint32_t i)
		{
			uint32_t idx = mixOdds[i];
			RealScalar p = exp((beta(idx+1)-beta(idx))*2.0*
					sv_[idx+1].logRatioRe(sv_[idx]));
			RealScalar u = urd.local()(re_.local());
			if(u < p)
			{
				std::swap(sv_[idx+1],sv_[idx]);
			}
		});
	}

	void sweep()
	{
		using std::real;
		tbb::parallel_for(0u, nTmp_*nChainsPer_,
			[&](uint32_t idx)
		{
			sweeper_.sweep(sv_[idx], beta(idx), re_.local());
		});
	}

	template<typename Container>
	void appendData(Container& container)
	{
		tbb::parallel_for(0u, nTmp_*nChainsPer_, nTmp_, 
			[&](uint32_t idx)
		{
			container.emplace_back(sv_[idx].data());
		});
	}
	
	/**
	 * \brief Sample using the MCMC.
	 * \return a container that contains data of samples.
	 * Total number of samples is nChainsPer_*nSweeps.
	 */
	auto sampling(uint32_t nSweeps, uint32_t nDiscard)
	{
		using std::norm;
		using std::pow;
		using std::abs;
		using std::exp;

		//Thermalizing phase
		for(uint32_t n = 0; n < nDiscard; n++)
		{
			sweep();
			mixChains();
		}
		using DataT = typename std::result_of<decltype(&StateValue::data)(StateValue)>::type;

		tbb::concurrent_vector<DataT> res;
		res.reserve(nSweeps);
		for(uint32_t ll = 0; ll < nSweeps; ll++)
		{
			sweep();
			mixChains();
			appendData(res);
		}
		return res;
	}
};
}
#endif//NNQS_SAMPLERS_SAMPLERPT_HPP
