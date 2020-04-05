#ifndef YANNQ_GROUNDSTATE_SRMATEACT_HPP
#define YANNQ_GROUNDSTATE_SRMATEACT_HPP
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 

#include <omp.h>

#include "ED/ConstructSparseMat.hpp"
#include "Utilities/Utility.hpp"
#include "Observables/ConstructDeltaExact.hpp"

namespace yannq
{
//! \addtogroup GroundState

//! \ingroup GroundState
//! This class calculate the quantum Fisher matrix by exactly constructing the quantum state.
template<typename Machine>
class SRMatExact
{
public:
	using ScalarType = typename Machine::ScalarType;
	using RealScalarType = typename remove_complex<ScalarType>::type;

	using MatrixType = typename Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorType = typename Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

private:
	const uint32_t n_;
	const Machine& qs_;
	tbb::concurrent_vector<uint32_t> basis_;

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
		
		deltas_ = constructDeltaExact(qs_, basis_);

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
		tbb::parallel_do(basis.begin(), basis.end(), 
				[&](uint32_t elt)
		{
			basis_.emplace_back(elt);
		});
		tbb::parallel_sort(basis_.begin(), basis_.end());
		ham_ = edp::constructSubspaceMat<double>(std::forward<ColFunc>(col), basis_);
	}
};
} //namespace yannq


#endif//YANNQ_GROUNDSTATE_SRMATEACT_HPP
