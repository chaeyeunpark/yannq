#ifndef CY_SRMATEXACT_HPP
#define CY_SRMATEXACT_HPP
#include <Eigen/Dense>

#include "ED/NodeMV.hpp"
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

	edp::NodeMV mv_;

	Vector st_;
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
		st_ = getPsi(qs_, true);
		deltas_.setZero(1<<n_,qs_.getDim());

		Vector k(1<<n_);
		mv_.perform_op(st_.data(), k.data()); //k = Ham*st_;
		energy_ = real(st_.adjoint()*k);
		
		Vector eloc = k.cwiseQuotient(st_);


#pragma omp parallel for schedule(static,8)
		for(uint32_t k = 0; k < (1u << n_); k++)
		{
			auto der = qs_.logDeriv(qs_.makeData(toSigma(n_, k)));
			deltas_.row(k) = der;
		}
		deltasPsis_ = st_.cwiseAbs2().asDiagonal()*deltas_; 
		grad_ = (deltas_*st_.asDiagonal()).adjoint()*k;
		grad_ -= energy_*deltasPsis_.colwise().sum();
	}

	Matrix corrMat() const
	{
		Matrix res = deltas_.adjoint()*deltasPsis_;
		auto t = deltasPsis_.colwise().sum();
		res -= t.adjoint()*t;
	}

	Vector getF() const
	{
		return grad_;
	}

	template<class ColFunc>
	SRMatExact(Machine& qs, ColFunc&& col)
	  : n_{qs.getN()}, qs_(qs), mv_(1<<n_, 0, 1<<n_, std::forward<ColFunc>(col))
	{
	}
};
} //namespace nnqs


#endif//CY_SRMATEXACT_HPP
