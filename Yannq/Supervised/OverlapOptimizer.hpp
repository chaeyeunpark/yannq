#ifndef YANNQ_SUPERVISED_OVERLAPOPTIMZER_HPP
#define YANNQ_SUPERVISED_OVERLAPOPTIMZER_HPP
#include <Eigen/Dense>
#include "Machines/RBM.hpp"

namespace yannq
{
template<class Machine, class Target>
class OverlapOptimizer
{
public:
	using ScalarType = typename Machine::ScalarType;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;

private:
	int N_;

	const Machine& qs_;
	const Target& target_;

	Matrix dervs_;
	Vector ratios_;
	ScalarType ov_;

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

#pragma omp parallel for schedule(static,8)
		for(std::size_t n = 0; n < rs.size(); n++)
		{
			const auto elt = rs[n];
			auto smp = construct_state<typename MachineStateTypes<Machine>::StateRef>(qs_, elt);

			dervs_.row(n) = qs_.logDeriv(elt);
			ratios_(n) = target_(toValue(smp.getSigma()))/qs_.coeff(elt);
		}
		ov_ = ratios_.mean();
	}

	Vector calcGrad() const
	{
		Vector res = dervs_.colwise().mean();
		res = res.conjugate();
		res -= dervs_.adjoint()*ratios_/ov_;
		return res;
	}
	
	//<\phi|\psi>
	ScalarType getOverlap() const
	{
		return ov_;
	}
};
}//namespace yannq
#endif//YANNQ_SUPERVISED_OVERLAPOPTIMZER_HPP
