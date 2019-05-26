#ifndef CY_OVERLAP_OPTIMIZER_HPP
#define CY_OVERLAP_OPTIMIZER_HPP
#include <Eigen/Dense>
#include "Machines/RBM.hpp"


namespace yannq
{
template<class Machine>
class OverlapOptimizer
{
public:
	using Scalar = typename Machine::ScalarType;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;

private:
	int N_;

	Machine& qs_;

	Eigen::VectorXcd target_;

	Matrix dervs_;
	Vector ratios_;

public:
	explicit OverlapOptimizer(Machine& qs, const Eigen::VectorXcd& target)
		: N_(qs.getN()), qs_(qs), target_(target)
	{
	}

	template<class Machine, class SamplingResult>
	void constructFromSampling(const Machine& qs, SamplingResult&& rs)
	{
		using std::conj;

		dervs_.resize(rs.size(), rbm.getDim());
		ratios_.resize(rs.size());

		dervs.setZero();
		ratios.setZero();

#pragma omp parallel for schedule(static,8)
		for(std::size_t n = 0; n < rs.size(); n++)
		{
			const auto elt = rs[n];
			auto smp = construct_state<typename MachineStateTypes<Machine>::StateRef>(qs, elt);

			dervs.row(n) = qs.logDeriv(elt);
			ratios(n) = target_(toValue(std::get<0>(elt)))/qs.coeff(elt);
		}
		ov_ = ratios_.mean();
	}

	Vector getGrad() const
	{
		Vector res = dervs_.colwise().mean();
		res -= dervs_.adjoint()*ratios/ov_;
		return res;
	}
	
	//<\phi|\psi>
	T getOverlap() const
	{
		return ov_;
	}
		/*
		Eigen::VectorXd rr = Eigen::Map<Eigen::VectorXd>((double*)res.data(), 2*rbm.getDim(), 1);
		return rr;
		*/
	/*
	double logOverlap() const
	{
		using std::real;
		typename yannq::RBM<T>::Vector psi = getPsi(rbm, true);
		T r = psi.adjoint()*target_;
		return -2*real(std::log(r));
	}
	*/
};
}//namespace yannq
#endif//CY_OVERLAP_OPTIMIZER_HPP
