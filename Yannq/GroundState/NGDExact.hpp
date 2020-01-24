#ifndef YANNQ_GROUNDSTATE_NGDEXACT_HPP
#define YANNQ_GROUNDSTATE_NGDEXACT_HPP
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <complex>

#include "ED/ConstructSparseMat.hpp"
#include "Utilities/Utility.hpp"

#include "Machines/AmplitudePhase.cpp"

namespace yannq
{

class NGDExact
{
public:
	using ReScalarType = AmplitudePhase::ScalarType;
	using CxScalarType = std::complex<ReScalarType>;

	using ReMatrixType = typename Eigen::Matrix<ReScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using ReVectorType = typename Eigen::Matrix<ReScalarType, Eigen::Dynamic, 1>;

	using CxMatrixType = typename Eigen::Matrix<CxScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using CxVectorType = typename Eigen::Matrix<CxScalarType, Eigen::Dynamic, 1>;
private:
	const uint32_t n_;
	const AmplitudePhase& qs_;
	std::vector<uint32_t> basis_;

	Eigen::SparseMatrix<ReScalarType> ham_;

	ReMatrixType deltas_;

	CxMatrixType deltasPsis_;
	CxMatrixType oloc_;
	ReVectorType grad_;

	ReScalarType energy_;

public:

	ReScalarType getEnergy() const
	{
		return energy_;
	}

	void constructExact()
	{
		deltas_.setZero(basis_.size(),qs_.getDim());

		CxVectorType st = getPsi(qs_, basis_, true);
		CxVectorType k = ham_*st;

		auto dimAmp = qs_.getDimAmp();
		auto dimPhase = qs_.getDimPhase();

		energy_ = std::real(CxScalarType(st.adjoint()*k));
		
#pragma omp parallel for schedule(static,8)
		for(uint32_t k = 0; k < basis_.size(); k++)
		{
			auto data = qs_.makeData(toSigma(n_, basis_[k]));
			deltas_.block(k, 0, 1, dimAmp) = qs_.logDerivAmp(data);
			deltas_.block(k, dimAmp, 1, dimPhase) = qs_.logDerivPhase(data);
		}
		deltasPsis_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		oloc_ = deltasPsis_.colwise().sum();
		deltasPsis_.rowwise() -= oloc_;
		
		grad_.resize(qs_.getDim());
		grad_.head(qs_.getDimAmp()) = 2.0*(k.adjoint()*deltasPsis_.leftCols(dimAmp)).real();
		grad_.tail(qs_.getDimPhase()) = 2.0*(k.adjoint()*deltasPsis_.rightCols(dimPhase)).imag();
	}

	CxVectorType oloc() const
	{
		return oloc_;
	}

	ReMatrixType corrMatAmp() const
	{
		auto dimAmp = qs_.getDimAmp();
		return deltas_.leftCols(dimAmp).transpose()*(deltasPsis_.leftCols(dimAmp)).real();
	}
	ReMatrixType corrMatPhase() const
	{
		auto dimPhase = qs_.getDimPhase();
		return deltas_.rightCols(dimPhase).transpose()*(deltasPsis_.rightCols(dimPhase)).imag();
	}

	ReVectorType getF() const
	{
		return grad_;
	}

	template<class ColFunc, class Iterable>
	NGDExact(const AmplitudePhase& qs, Iterable&& basis, ColFunc&& col)
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
#endif//YANNQ_GROUNDSTATE_NGDEXACT_HPP
