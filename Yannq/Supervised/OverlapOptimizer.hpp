#ifndef YANNQ_SUPERVISED_OVERLAPOPTIMZER_HPP
#define YANNQ_SUPERVISED_OVERLAPOPTIMZER_HPP
#include <Eigen/Dense>

#include <tbb/tbb.h>

#include "Machines/RBM.hpp"

namespace yannq
{
template<class Machine, class Target>
class OverlapOptimizer
{
public:
	using Scalar = typename Machine::Scalar;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;

private:
	int N_;

	const Machine& qs_;
	const Target& target_;

	Matrix dervs_;
	Vector ratios_;
	Scalar ov_;

public:
	explicit OverlapOptimizer(const Machine& qs, const Target& target)
		: N_(qs.getN()), qs_(qs), target_(target)
	{
	}

	template<class SamplingResult>
	void constructFromSampling(SamplingResult&& rs)
	{
		using std::conj;

		dervs_.resize(rs.size(), qs_.getDim());
		ratios_.resize(rs.size());

		dervs_.setZero();
		ratios_.setZero();

		tbb::parallel_for(0u, rs.size(), 
			[&](uint32_t idx)
		{
			const auto elt = rs[idx];
			auto smp = construct_state<typename MachineStateTypes<Machine>::StateRef>(qs_, elt);

			dervs_.row(idx) = qs_.logDeriv(elt);
			ratios_(idx) = target_(toValue(smp.getSigma()))/qs_.coeff(elt);
		});
		ov_ = ratios_.mean();
	}

	Vector calcGrad() const
	{
		Vector res = dervs_.colwise().mean();
		res = res.conjugate();
		res -= dervs_.adjoint()*ratios_/ov_/dervs_.rows();
		return res;
	}
	
	//<\phi|\psi>/Z
	Scalar meanRatio() const
	{
		return ov_;
	}
	Scalar meanRatioSqr() const
	{
		return ratios_.cwiseAbs2().mean();
	}
};
}//namespace yannq
#endif//YANNQ_SUPERVISED_OVERLAPOPTIMZER_HPP
