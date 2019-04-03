#ifndef CY_HESSIANEXACT_HPP
#define CY_HESSIANEXACT_HPP
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ED/ConstructSparseMat.hpp"
#include "Utilities/Utility.hpp"

namespace nnqs
{

template<typename Machine>
class HessianExact
{
public:
	using Scalar = typename Machine::ScalarType;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Machine::Matrix;
	using Vector = typename Machine::Vector;
private:
	int n_;
	Machine& qs_;

	Eigen::SparseMatrix<RealScalar> ham_;

	Matrix deltas_;
	Matrix deltasPsis_;

	Matrix eDelta_;

	Vector eloc_;
	Vector grad_;
	Vector eDer_; // <\psi |H|\del_i \psi> /<\psi|\psi>

	Scalar energy_;

public:

	RealScalar getEnergy() const
	{
		return real(energy_);
	}

	void constructExact()
	{
		Vector st = getPsi(qs_, false);
		double Z = st.squaredNorm();
		deltas_.setZero(1<<n_,qs_.getDim());

		energy_ = st.adjoint()*k;
		
		eloc_ = k.cwiseQuotient(st);

#pragma omp parallel for schedule(static,8)
		for(uint32_t k = 0; k < (1u << n_); k++)
		{
			auto der = qs_.logDeriv(qs_.makeData(toSigma(n_, k)));
			deltas_.row(k) = der;
		}
		ham_*(st.asDiagonal()*deltas_)
		deltasPsis_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		grad_ = (st.asDiagonal()*deltas_).adjoint()*k;
		grad_ -= t*deltasPsis_.colwise().sum().conjugate();
	}

	Vector deltaMean() const
	{
		return deltasPsis_.colwise().sum();
	}

	Matrix hessian() const
	{
		Matrix res1 = deltas_.adjoint()*deltasPsis_;
	}

	Vector getF() const
	{
		return grad_;
	}

	template<class ColFunc>
	HessianExact(Machine& qs, ColFunc&& col)
	  : n_{qs.getN()}, qs_(qs), ham_(edp::constructSparseMat<RealScalar>(1<<n_, std::forward<ColFunc>(col)))
	{
	}
};
} //namespace nnqs


#endif//CY_HESSIANEXACT_HPP
