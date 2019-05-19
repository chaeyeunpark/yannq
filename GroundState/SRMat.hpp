#ifndef YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
#define YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "Utilities/Utility.hpp"

namespace yannq
{

template<typename Machine>
class SRMat;

} //namespace yannq

namespace Eigen {
namespace internal {
	template<typename Machine>
	struct traits<yannq::SRMat<Machine> > :  public Eigen::internal::traits<Eigen::SparseMatrix<typename Machine::ScalarType> > {};
}
} //namespace Eigen




namespace yannq
{
/** 
 * Construct the fisher information metric for quantum states
 * */
template<typename Machine>
class SRMat
	: public Eigen::EigenBase<SRMat<Machine> > 
{
public:
	using Scalar = typename Machine::ScalarType;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Machine::Matrix;
	using Vector = typename Machine::Vector;
private:
	int n_;

	Machine& qs_;
	RealScalar shift_;
	

	double eloc_;

	double elocVar_;


	Matrix deltas_;
	Vector deltaMean_;
	Vector grad_;
	Vector elocs_;

public:
	// Required typedefs, constants, and method:
	typedef int StorageIndex;

	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = false
	};

	Eigen::Index rows() const { return qs_.getDim(); }
	Eigen::Index cols() const { return qs_.getDim(); }

	template<typename Rhs>
	Eigen::Product<SRMat<Machine>, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
	  return Eigen::Product<SRMat<Machine>, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
	}

	void setShift(RealScalar shift)
	{
		shift_ = shift;
	}

	RealScalar getShift() const
	{
		return shift_;
	}

	template<class SamplingResult, class Hamiltonian>
	void constructFromSampling(SamplingResult& rs, Hamiltonian& ham)
	{
		int nsmp = rs.size();

		deltas_.setZero(nsmp, qs_.getDim());
		grad_.setZero(nsmp);
		elocs_.setZero(nsmp);

#pragma omp parallel for schedule(static,8)
		for(std::size_t n = 0; n < rs.size(); n++)
		{
			const auto& elt = rs[n];
			auto smp = construct_state<typename MachineStateTypes<Machine>::StateRef>(qs_, elt);

			elocs_(n) = ham(smp);

			deltas_.row(n) = qs_.logDeriv(elt);
		}
		deltaMean_ = deltas_.colwise().mean();
		deltas_ = deltas_.rowwise() - deltaMean_.transpose();

		eloc_ = real(elocs_.mean());
		elocVar_ = elocs_.real().cwiseAbs2().sum()/rs.size() - eloc_*eloc_;

		elocs_ -= elocs_.mean() * Eigen::VectorXd::Ones(nsmp);

		grad_ = deltas_.adjoint() * elocs_;
		grad_ /= nsmp;

	}
	Vector oloc() const
	{
		return deltaMean_;
	}

	Matrix corrMat() const
	{
		int nsmp = deltas_.rows();
		return (deltas_.adjoint() * deltas_)/nsmp;
	}

	double getEloc() const
	{
		return eloc_;
	}

	double getElocVar() const
	{
		return elocVar_;
	}
	
	SRMat(Machine& qs)
	  : n_{qs.getN()}, qs_(qs), shift_{}
	{
	}

	Vector getF() const
	{
		return grad_;
	}

	template<class Rhs>
	typename Machine::Vector apply(const Rhs& rhs) const
	{
		assert(rhs.size() == qs_.getDim());
		typename Machine::Vector r = deltas_*rhs;

		typename Machine::Vector res = deltas_.adjoint()*r/r.rows();

		return res + Scalar(shift_)*rhs;
	}

};
} //namespace yannq

// Implementation of yannq::SRMat * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
	template<typename Rhs, typename Machine>
	struct generic_product_impl<yannq::SRMat<Machine>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
	: generic_product_impl_base<yannq::SRMat<Machine>, Rhs, generic_product_impl<yannq::SRMat<Machine>, Rhs> >
	{
		typedef typename Product<yannq::SRMat<Machine>, Rhs>::Scalar Scalar;
		template<typename Dest>
		static void scaleAndAddTo(Dest& dst, const yannq::SRMat<Machine>& lhs, const Rhs& rhs, const Scalar& alpha)
		{
			// This method should implement "dst += alpha * lhs * rhs" inplace,
			// however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
			assert(alpha==Scalar(1) && "scaling is not implemented");
			EIGEN_ONLY_USED_FOR_DEBUG(alpha);

			dst += lhs.apply(rhs);
		}
	};
} //namespace internal
} //namespace Eigen

#endif//YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
