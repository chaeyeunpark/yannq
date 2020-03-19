#ifndef YANNQ_GROUNDSTATE_SRMATEACT_HPP
#define YANNQ_GROUNDSTATE_SRMATEACT_HPP
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 

#include <omp.h>

#include "ED/ConstructSparseMat.hpp"
#include "Utilities/Utility.hpp"

namespace yannq
{
//! \addtogroup GroundState

//! \ingroup GroundState
//! This class calculate the quantum Fisher matrix by exactly constructing the quantum state.
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
	VectorType oloc_;
	VectorType grad_;

	RealScalarType energy_;
	RealScalarType energyVar_;

public:

	RealScalarType eloc() const
	{
		return energy_;
	}

	RealScalarType elocVar() const
	{
		return energyVar_;
	}

	void constructExact()
	{
		VectorType st = getPsi(qs_, basis_, true);

		VectorType k = ham_*st;

		std::complex<double> t = st.adjoint()*k;
		energy_ = std::real(t);
		energyVar_ = static_cast<std::complex<double> >(k.adjoint()*k).real();
		energyVar_ -= energy_*energy_;
		
		deltas_.setZero(basis_.size(),qs_.getDim());
#pragma omp parallel
		{
			MatrixType local(8, qs_.getDim());

#pragma omp for schedule(dynamic)
			for(uint32_t k = 0; k < basis_.size(); k+=8)
			{
				for(int l = 0; l < 8; ++l)
				{
					local.row(l) = qs_.logDeriv(qs_.makeData(toSigma(n_, basis_[k+l])));
				}
				deltas_.block(k, 0, 8, qs_.getDim()) = local;
			}
		}
		deltasPsis_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		oloc_ = deltasPsis_.colwise().sum();
		grad_ = (st.asDiagonal()*deltas_).adjoint()*k;
		grad_ -= t*oloc_.conjugate();
	}

	const VectorType& oloc() const&
	{
		return oloc_;
	}
	VectorType oloc() &&
	{
		return oloc_;
	}

	MatrixType corrMat() const
	{
		MatrixType res = deltas_.adjoint()*deltasPsis_;
		res -= oloc_.conjugate()*oloc_.transpose();
		return res;
	}

	const VectorType& energyGrad() const&
	{
		return grad_;
	}
	VectorType erengyGrad() &&
	{
		return grad_;
	}

	VectorType apply(const VectorType& rhs)
	{
		VectorType res = deltas_.adjoint()*(deltasPsis_*rhs);
		res -= oloc_.conjugate()*(oloc_.transpose()*rhs);
		return res;
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
