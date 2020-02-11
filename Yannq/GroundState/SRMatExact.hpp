#ifndef YANNQ_GROUNDSTATE_SRMATEACT_HPP
#define YANNQ_GROUNDSTATE_SRMATEACT_HPP
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 


#include "ED/ConstructSparseMat.hpp"
#include "Utilities/Utility.hpp"

namespace yannq
{

template<typename Machine, typename ScalarType = typename Machine::ScalarType>
class SRMatExact
{
public:
	using RealScalarType = typename remove_complex<ScalarType>::type;

	using MatrixType = typename Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorType = typename Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

private:
	const uint32_t n_;
	const Machine& qs_;
	std::vector<uint32_t> basis_;

	Eigen::SparseMatrix<RealScalarType> ham_;

	MatrixType deltas_;
	MatrixType deltasPsis_;
	VectorType grad_;

	RealScalarType energy_;

public:

	RealScalarType getEnergy() const
	{
		return energy_;
	}

	void constructExact()
	{
		VectorType st = getPsi(qs_, basis_, true);
		deltas_.setZero(basis_.size(),qs_.getDim());

		VectorType k = ham_*st;

		std::complex<double> t = st.adjoint()*k;
		energy_ = std::real(t);
		
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

	VectorType oloc() const
	{
		return deltasPsis_.colwise().sum();
	}

	MatrixType corrMat() const
	{
		MatrixType res = deltas_.adjoint()*deltasPsis_;
		auto t = deltasPsis_.colwise().sum();
		res -= t.adjoint()*t;
		return res;
	}

	VectorType getF() const
	{
		return grad_;
	}


	template<class Iterable, class ColFunc>
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
