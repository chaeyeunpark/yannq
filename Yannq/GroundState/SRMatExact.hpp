#ifndef YANNQ_GROUNDSTATE_SRMATEACT_HPP
#define YANNQ_GROUNDSTATE_SRMATEACT_HPP
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ED/ConstructSparseMat.hpp"
#include "Utilities/Utility.hpp"

namespace yannq
{

template<typename Machine>
class SRMatExact
{
public:
	using Scalar = typename Machine::ScalarType;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Machine::Matrix;
	using Vector = typename Machine::Vector;
private:
	const int n_;
	const Machine& qs_;
	std::vector<uint32_t> basis_;

	Eigen::SparseMatrix<RealScalar> ham_;

	Matrix deltas_;
	Matrix deltasPsis_;
	Vector grad_;

	RealScalar energy_;

public:

	RealScalar getEnergy() const
	{
		return energy_;
	}

	void constructExact()
	{
		Vector st = getPsi(qs_, basis_, true);
		deltas_.setZero(basis_.size(),qs_.getDim());

		Vector k = ham_*st;

		std::complex<double> t = st.adjoint()*k;
		energy_ = std::real(t);
		
		Vector eloc = k.cwiseQuotient(st);


#pragma omp parallel for schedule(static,8)
		for(uint32_t k = 0; k < basis_.size(); k++)
		{
			auto der = qs_.logDeriv(qs_.makeData(toSigma(n_, basis_[k])));
			deltas_.row(k) = der;
		}
		deltasPsis_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		grad_ = (st.asDiagonal()*deltas_).adjoint()*k;
		grad_ -= t*deltasPsis_.colwise().sum().conjugate();
	}

	Vector oloc() const
	{
		return deltasPsis_.colwise().sum();
	}

	Matrix corrMat() const
	{
		Matrix res = deltas_.adjoint()*deltasPsis_;
		auto t = deltasPsis_.colwise().sum();
		res -= t.adjoint()*t;
		return res;
	}

	Vector getF() const
	{
		return grad_;
	}

	template<class ColFunc, class Iterable>
	SRMatExact(const Machine& qs, Iterable&& basis, ColFunc&& col)
	  : n_{qs.getN()}, qs_(qs)
	{
		for(auto elt: basis)
		{
			basis_.emplace_back(elt);
		}
		ham_ = edp::constructSubspaceMat<double>(std::forward<ColFunc>(col), basis_);
	}
};
} //namespace yannq


#endif//YANNQ_GROUNDSTATE_SRMATEACT_HPP
