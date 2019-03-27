#ifndef CY_SRMATEXACT_HPP
#define CY_SRMATEXACT_HPP
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ED/ConstructSparseMat.hpp"
#include "Utilities/Utility.hpp"

namespace nnqs
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
	int n_;
	Machine& qs_;

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
		Vector st = getPsi(qs_, true);
		deltas_.setZero(1<<n_,qs_.getDim());

		Vector k = ham_*st;

		std::complex<double> t = st.adjoint()*k;
		energy_ = std::real(t);
		
		Vector eloc = k.cwiseQuotient(st);


#pragma omp parallel for schedule(static,8)
		for(uint32_t k = 0; k < (1u << n_); k++)
		{
			auto der = qs_.logDeriv(qs_.makeData(toSigma(n_, k)));
			deltas_.row(k) = der;
		}
		deltasPsis_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		grad_ = (st.asDiagonal()*deltas_).adjoint()*k;
		grad_ -= t*deltasPsis_.colwise().sum().conjugate();
	}

	Vector deltaMean() const
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

	template<class ColFunc>
	SRMatExact(Machine& qs, ColFunc&& col)
	  : n_{qs.getN()}, qs_(qs), ham_(edp::constructSparseMat<RealScalar>(1<<n_, std::forward<ColFunc>(col)))
	{
	}
};
} //namespace nnqs


#endif//CY_SRMATEXACT_HPP
