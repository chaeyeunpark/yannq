#ifndef YANNQ_OBSERVARBLES_CONSTRUCTDELTAEXACT_HPP
#define YANNQ_OBSERVARBLES_CONSTRUCTDELTAEXACT_HPP
#include <Eigen/Dense>

#include <tbb/tbb.h>

template<class Machine, class RandomIterable>
typename Machine::MatrixType constructDeltaExact(const Machine& qs, RandomIterable&& basis)
{
	using MatrixType = typename Machine::MatrixType;
	using Range = tbb::blocked_range<std::size_t>;
	const int N = qs.getN();
	MatrixType deltas(basis.size(), qs.getDim());
	deltas.setZero(basis.size(), qs.getDim());
	if(basis.size() >= 32)
	{
		tbb::parallel_for(Range(std::size_t(0u), basis.size(), 8),
			[&](const Range& r)
		{
			MatrixType tmp(r.end()-r.begin(), qs.getDim());
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

#endif//YANNQ_OBSERVARBLES_CONSTRUCTDELTAEXACT_HPP
