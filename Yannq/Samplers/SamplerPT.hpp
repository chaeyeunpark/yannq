#ifndef NNQS_SAMPLERS_SAMPLERPT_HPP
#define NNQS_SAMPLERS_SAMPLERPT_HPP
#include <vector>
#include <random>
#include <complex>
#include <nlohmann/json.hpp>

#include <tbb/tbb.h>

namespace yannq
{

template<class Machine, class RandomEngine, class StateValue, class Sweeper>
class SamplerPT
{
public:
	using Scalar = typename Machine::Scalar;
	using RealScalar = typename Machine::RealScalar;

private:
	const Machine& qs_;
	const uint32_t n_;
	const uint32_t numChain_;
	std::vector<RealScalar> betas_;

	std::vector<RandomEngine> re_;
	Sweeper& sweeper_;

	std::vector<StateValue> sv_;

public:
	SamplerPT(const Machine& qs, uint32_t numChain, Sweeper& sweeper)
		: qs_(qs), n_(qs.getN()), numChain_(numChain),
		re_(numChain, RandomEngine{}),
		sweeper_(sweeper)
	{
		for(uint32_t idx = 0; idx < numChain; idx++)
		{
			betas_.emplace_back( RealScalar(numChain - idx)/numChain );
		}
	}

	nlohmann::json desc() const
	{
		nlohmann::json res;
		res["name"] = "SamplerPT";
		res["num_chains"] = numChain_;
		res["sweeper"] = sweeper_.name();
		return res;
	}

	void initializeRandomEngine()
	{
		std::random_device rd;
		for(uint32_t idx = 0u; idx < numChain_; ++idx)
		{
			re_[idx].seed(rd());
		}
	}
	
	void randomizeSigma()
	{
		sv_.clear();
		for(uint32_t idx = 0u; idx < numChain_; ++idx)
		{
			sv_.emplace_back(qs_, randomSigma(n_, re_[idx]));
		}
	}

	void randomizeSigma(int nup)
	{
		sv_.clear();
		for(uint32_t idx = 0u; idx < numChain_; ++idx)
		{
			sv_.emplace_back(qs_, randomSigma(n_, nup, re_[idx]));
		}
	}

	void mixChains()
	{
		using std::real;
		tbb::enumerable_thread_specific<
			std::uniform_real_distribution<RealScalar> > urd(0.0,1.0);

		tbb::parallel_for(0u, numChain_, 2u,
			[&](uint32_t idx)
		{
			RealScalar p = exp((betas_[idx+1]-betas_[idx])*2.0*
					sv_[idx+1].logRatioRe(sv_[idx]));
			RealScalar u = urd.local()(re_[idx]);
			if(u < p)
			{
				std::swap(sv_[idx+1],sv_[idx]);
			}
		});
		tbb::parallel_for(1u, numChain_-1, 2u, 
			[&](uint32_t idx)
		{
			RealScalar p = exp((betas_[idx+1]-betas_[idx])*2.0*
					sv_[idx+1].logRatioRe(sv_[idx]));
			RealScalar u = urd.local()(re_[idx]);
			if(u < p)
			{
				std::swap(sv_[idx+1],sv_[idx]);
			}
		});
	}

	void sweep()
	{
		using std::real;
		tbb::parallel_for(0u, numChain_,
			[&](uint32_t idx)
		{
			sweeper_.localSweep(sv_[idx], betas_[idx], re_[idx]);
		});
	}


	auto sampling(uint32_t n_sweeps, uint32_t n_therm)
	{
		using std::norm;
		using std::pow;
		using std::abs;
		using std::exp;

		//Thermalizing phase
		for(uint32_t n = 0; n < n_therm; n++)
		{
			sweep();
			mixChains();
		}
		using DataT = typename std::result_of<decltype(&StateValue::data)(StateValue)>::type;

		std::vector<DataT> res;
		res.reserve(n_sweeps);
		for(uint32_t ll = 0; ll < n_sweeps; ll++)
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
