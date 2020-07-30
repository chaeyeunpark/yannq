#pragma once
#include <Eigen/Dense>

#include <tbb/tbb.h>

#include "Utilities/Utility.hpp"
#include "Observables/Observable.hpp"

namespace yannq
{
template<class Machine, class SamplingResult>
typename Machine::Matrix constructDelta(const Machine& qs, SamplingResult&& sr)
{
	using Matrix = typename Machine::Matrix;
	using Range = tbb::blocked_range<std::size_t>;

	Matrix deltas(sr.size(), qs.getDim());
	deltas.setZero(sr.size(), qs.getDim());
	if(sr.size() >= 32)
	{
		tbb::parallel_for(Range(std::size_t(0u), sr.size(), 8),
			[&](const Range& r)
		{
			Matrix tmp(r.end()-r.begin(), qs.getDim());
			uint32_t start = r.begin();
			uint32_t end = r.end();
			for(uint32_t l = 0; l < end-start; ++l)
			{
				tmp.row(l) = qs.logDeriv(sr[l+start]);
				//qs.makeData(toSigma(N, basis[l+start]))
			}
			deltas.block(start, 0, end-start, qs.getDim()) = tmp;
		}, tbb::simple_partitioner());
	}
	else
	{
		for(uint32_t k = 0; k < sr.size(); k++)
		{
			deltas.row(k) = qs.logDeriv(sr[k]);
		}
	}
	return deltas;
}

namespace detail
{

template<typename Derived>
void initObs(uint32_t nsmp, Observable<Derived>& obs)
{
	obs.initIter(nsmp);
}
template<typename Derived, typename ...Derived2>
void initObs(uint32_t nsmp, Observable<Derived>& obs, Observable<Derived2>&... args)
{
	initObs(nsmp, obs);
	initObs(nsmp, args...);
}

template<typename State, typename Derived>
void eachSampleObs(std::size_t idx, State&& state, Observable<Derived>& obs)
{
	obs.eachSample(idx, std::forward<State>(state));
}

template<typename State, typename Derived, typename ...Derived2>
void ecahSampleObs(size_t idx, State&& state,
		Observable<Derived>& obs, Observable<Derived2>&... args)
{
	eachSampleObs(idx, std::forward<State>(state), obs);
	eachSampleObs(idx, std::forward<State>(state), args...);
}
template<typename Derived>
void finObs(Observable<Derived>& obs)
{
	obs.finIter();
}
template<typename Derived, typename ...Derived2>
void findObs(Observable<Derived>& obs, Observable<Derived2>&... args)
{
	finObs(obs);
	finObs(args...);
}

template<typename RealVector, typename Derived>
void finObsWeights(const Eigen::Ref<RealVector>& weights, Observable<Derived>& obs)
{
	obs.finIter();
}
template<typename RealVector, typename Derived, typename ...Derived2>
void findObsWeights(const Eigen::Ref<RealVector>& weights,
		Observable<Derived>& obs, Observable<Derived2>&... args)
{
	finObsWeights(weights, obs);
	finObsWeights(weights, args...);
}

} //namspeace yannq::detail

template<class Machine, class SamplingResult, typename ...Derived>
void constructObs(const Machine& qs, SamplingResult&& sr, Observable<Derived>&... obs)
{
	using Matrix = typename Machine::Matrix;
	using Range = tbb::blocked_range<std::size_t>;
	uint32_t nsmp = sr.size();
	detail::initObs(nsmp, obs...);

	tbb::parallel_for(std::size_t(0u), sr.size(),
		[&](std::size_t idx)
	{
		const auto& elt = sr[idx];
		auto state = construct_state
				<typename MachineStateTypes<Machine>::StateRef>(qs, elt);
		detail::eachSampleObs(idx, state, obs...);
	});

	detail::finObs(obs...);
}
template<class Machine, class SamplingResult, typename ...Derived>
void constructObsWeights(const Machine& qs, SamplingResult&& sr, 
		const Eigen::Ref<const typename Machine::RealVector>& weights,
		Observable<Derived>&... obs)
{
	using Matrix = typename Machine::Matrix;
	using Range = tbb::blocked_range<std::size_t>;
	uint32_t nsmp = sr.size();
	detail::initObs(nsmp, obs...);

	tbb::parallel_for(std::size_t(0u), sr.size(),
		[&](std::size_t idx)
	{
		const auto& elt = sr[idx];
		auto state = construct_state
				<typename MachineStateTypes<Machine>::StateRef>(qs, elt);
		detail::eachSampleObs(idx, state, obs...);
	});

	detail::finObsWeights(weights, obs...);
}
} //namespace yannq
