#ifndef YANNQ_OBSERVARBLES_CONSTRUCTDELTAEXACT_HPP
#define YANNQ_OBSERVARBLES_CONSTRUCTDELTAEXACT_HPP
#include <Eigen/Dense>
template<class Machine, class RandomIterable>
typename Machine::MatrixType constructDeltaExact(const Machine& qs, RandomIterable&& basis)
{
	using MatrixType = typename Machine::MatrixType;
	const int N = qs.getN();
	MatrixType deltas(basis.size(), qs.getDim());
	deltas.setZero(basis.size(), qs.getDim());
	if(basis.size() > 32)
	{
#pragma omp parallel
		{
			MatrixType local(8, qs.getDim());

#pragma omp for schedule(dynamic)
			for(uint32_t k = 0; k < basis.size(); k+=8)
			{
				uint32_t togo = std::min(8u, static_cast<uint32_t>(basis.size()-k));
				for(int l = 0; l < togo; ++l)
				{
					local.row(l) = 
						qs.logDeriv(qs.makeData(toSigma(N, basis[k+l])));
				}
				deltas.block(k, 0, togo, qs.getDim()) = local.topRows(togo);
			}
		}
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
