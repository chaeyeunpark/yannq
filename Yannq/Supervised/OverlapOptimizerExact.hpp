#ifndef YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#define YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#include <Eigen/Dense>

#include "Utilities/Utility.hpp"

namespace yannq
{
template<typename Machine>
class OverlapOptimizerExact
{
public:
	using ScalarType = typename Machine::ScalarType;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;
private:
	int N_;
	Vector target_;

public:

	explicit OverlapOptimizerExact(int N, const Vector& target)
		: N_(N), target_(target)
	{
	}

	/**
	 * This method return the gradient of the lograithmic fidelity: -log(\langle \psi_\theta | \phi  \rangle).
	 * */
	typename Machine::Vector gradient(const Machine& qs) const
	{
		using std::conj;
		using Vector = typename Machine::Vector;

		Vector res(qs.getDim());
		Vector r1(qs.getDim());
		res.setZero();
		r1.setZero();

		Vector psi = getPsi(qs, false);
#pragma omp parallel
		{
			Vector resLocal(qs.getDim());
			Vector r1Local(qs.getDim());
			resLocal.setZero();
			r1Local.setZero();

#pragma omp for schedule(static,8)
			for(uint32_t n = 0; n < (1u<<N_); n++)
			{
				auto s = toSigma(N_, n);
				auto der = qs.logDeriv(qs.makeData(s));
				resLocal += der.conjugate()*std::norm(psi(n));
				r1Local += conj(psi(n))*target_(n)*der.conjugate();
			}
#pragma omp critical
			{
				res += resLocal;
				r1 += r1Local;
			}
		}
		std::complex<double> r = psi.adjoint() * target_;
		res -= r1/r;

		return res;
	}
	
	/**
	 * return squared fidelity: |\langle \psi_\theta | \phi \rangle|^2
	 */
	double fidelity(const Machine& qs) const
	{
		using std::norm;
		auto psi = getPsi(qs, false);
		return norm( psi.adjoint() * target_);
	}
};
} //yannq
#endif//YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
