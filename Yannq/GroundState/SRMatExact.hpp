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
	using Scalar = typename Machine::Scalar;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Vector = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

private:
	const uint32_t n_;
	const Machine& qs_;
	tbb::concurrent_vector<uint32_t> basis_;

	Eigen::SparseMatrix<RealScalar> ham_;

	Matrix deltas_;
	Matrix deltasPsis_;
	Vector oloc_;
	Vector grad_;

	RealScalar energy_;
	RealScalar energyVar_;

public:

	RealScalar eloc() const
	{
		return energy_;
	}

	RealScalar elocVar() const
	{
		return energyVar_;
	}

	void constructExact()
	{
		Vector st = getPsi(qs_, basis_, true);

		Vector k = ham_*st;

		Scalar t = st.adjoint()*k;
		energy_ = std::real(t);
		energyVar_ = static_cast<Scalar>(k.adjoint()*k).real();
		energyVar_ -= energy_*energy_;
		
		deltas_ = constructDeltaExact(qs_, basis_);

		deltasPsis_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		oloc_ = deltasPsis_.colwise().sum();
		grad_ = (st.asDiagonal()*deltas_).adjoint()*k;
		grad_ -= t*oloc_.conjugate();
	}

	const Vector& oloc() const&
	{
		return oloc_;
	}
	Vector oloc() &&
	{
		return oloc_;
	}

	Matrix corrMat() const
	{
		Matrix res = deltas_.adjoint()*deltasPsis_;
		res -= oloc_.conjugate()*oloc_.transpose();
		return res;
	}

	const Vector& energyGrad() const&
	{
		return grad_;
	}
	Vector erengyGrad() &&
	{
		return grad_;
	}

	Vector apply(const Vector& rhs)
	{
		Vector res = deltas_.adjoint()*(deltasPsis_*rhs);
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
		ham_ = edp::constructSubspaceMat<RealScalar>
			(std::forward<ColFunc>(col), basis_);
	}
};
} //namespace yannq


#endif//YANNQ_GROUNDSTATE_SRMATEACT_HPP
