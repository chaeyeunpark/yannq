#pragma once
#include <Eigen/Dense>

#include <tbb/tbb.h>

#include "Utilities/Utility.hpp"
namespace yannq
{
template<class Machine, class RandomIterable>
typename Machine::Matrix constructDeltaExact(const Machine& qs, RandomIterable&& basis)
{
	using Matrix = typename Machine::Matrix;
	using Range = tbb::blocked_range<std::size_t>;
	const int N = qs.getN();
	Matrix deltas(basis.size(), qs.getDim());
	deltas.setZero(basis.size(), qs.getDim());
	if(basis.size() >= 32)
	{
		tbb::parallel_for(Range(std::size_t(0u), basis.size(), 8),
			[&](const Range& r)
		{
			Matrix tmp(r.end()-r.begin(), qs.getDim());
			uint32_t start = r.begin();
			uint32_t end = r.end();
			for(int l = 0; l < end-start; ++l)
			{
				tmp.row(l) = 
					qs.logDeriv(qs.makeData(toSigma(N, basis[l+start])));
			}
			deltas.block(start, 0, end-start, qs.getDim()) = tmp;
		}, tbb::simple_partitioner());
	}
	else
	{
		for(uint32_t k = 0; k < basis.size(); k++)
		{
			deltas.row(k) = qs.logDeriv(qs.makeData(toSigma(N, basis[k])));
		}
	}
	return deltas;
}
} //namespace yannq
