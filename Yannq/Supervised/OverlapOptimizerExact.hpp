#ifndef YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#define YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#include <Eigen/Dense>
#include <limits>
#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

namespace yannq
{
template<typename Machine>
class OverlapOptimizerExact
{
public:
	using Scalar = typename Machine::Scalar;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;

private:
	const Machine& qs_;
	const int N_;
	Vector target_;

	Matrix deltas_;
	Matrix psiDeltas_;
	Vector ovs_;
	Vector oloc_;
	Scalar ov_;

public:
	template<class Target>
	explicit OverlapOptimizerExact(const Machine& qs, const Target& t)
		: qs_(qs), N_(qs.getN())
	{
		target_.resize(1<<N_);
		for(int i = 0; i < (1<<N_); i++)
		{
			target_(i) = t(i);
		}
	}

	Vector getTarget() const 
	{
		return target_;
	}

	/**
	 * This method return the gradient of the lograithmic fidelity: -log(\langle \psi_\theta | \phi  \rangle).
	 * */
	void constructExact() 
	{
		using std::conj;
		using Vector = typename Machine::Vector;

		deltas_.setZero(1<<N_, qs_.getDim());

		Vector st = getPsi(qs_, true);

#pragma omp parallel for schedule(static,8)
		for(uint32_t n = 0; n < (1u<<N_); n++)
		{
			auto s = toSigma(N_, n);
			deltas_.row(n) = qs_.logDeriv(qs_.makeData(s));
		}
		psiDeltas_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		ovs_ = st.conjugate().cwiseProduct(target_);
		ov_ = ovs_.sum();
		oloc_ = psiDeltas_.colwise().sum();
	}

	Vector calcGrad() const
	{
		Vector res = oloc().conjugate();
		Vector r1 = ovs_.transpose()*deltas_.conjugate();
		res -= r1/ov_;

		return res;
	}
	
	Vector oloc() const
	{
		return oloc_;
	}

	Matrix corrMat() const
	{
		Matrix res = deltas_.adjoint()*psiDeltas_;
		return res - oloc_.conjugate()*oloc_.transpose();
	}

	/**
	 * return squared fidelity: |\langle \psi_\theta | \phi \rangle|^2
	 */
	double fidelity() const
	{
		using std::norm;
		return norm(ov_);
	}
};
} //yannq
#endif//YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
