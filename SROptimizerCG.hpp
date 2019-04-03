#ifndef CY_SROPTIMIZER_CG_HPP
#define CY_SROPTIMIZER_CG_HPP
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "Utilities/Utility.hpp"

namespace nnqs
{

template<typename Machine>
class SRMatFree;

} //namespace nnqs

namespace Eigen {
namespace internal {
	template<typename Machine>
	struct traits<nnqs::SRMatFree<Machine> > :  public Eigen::internal::traits<Eigen::SparseMatrix<typename Machine::ScalarType> > {};
}
} //namespace Eigen

namespace nnqs
{

template<typename Machine>
class SRMatFree
	: public Eigen::EigenBase<SRMatFree<Machine> > 
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
	Eigen::Product<SRMatFree<Machine>, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
	  return Eigen::Product<SRMatFree<Machine>, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
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
		deltaMean_.setZero(nsmp);
		grad_.setZero(nsmp);

		Eigen::VectorXcd eloc(nsmp);

#pragma omp parallel for schedule(static,8)
		for(std::size_t n = 0; n < rs.size(); n++)
		{
			const auto& elt = rs[n];
			auto smp = construct_state<typename MachineStateTypes<Machine>::StateRef>(qs_, elt);

			eloc(n) = ham(smp);

			deltas_.row(n) = qs_.logDeriv(elt);
		}
		deltaMean_ = deltas_.colwise().mean();
		grad_ = deltas_.adjoint()*eloc/nsmp;
		grad_ -= eloc.mean()*deltaMean_.conjugate();
		eloc_ = real(eloc.mean());
		elocVar_ = eloc.real().cwiseAbs2().sum()/rs.size() - eloc_*eloc_;
	}

	Matrix corrMat() const
	{
		//const int dim = qs_.getDim();
		int nsmp = deltas_.rows();
		Matrix res = deltas_.adjoint()*deltas_/nsmp;
		res -= deltaMean_.conjugate()*deltaMean_.transpose();
		return res;// + shift_*Matrix::Identity(dim,dim);;
	}

	double getEloc() const
	{
		return eloc_;
	}

	double getElocVar() const
	{
		return elocVar_;
	}
	
	SRMatFree(Machine& qs)
	  : n_{qs.getN()}, qs_(qs), shift_{}
	{
	}
	Vector getF() const
	{
		return grad_;
	}
	const Eigen::MatrixXcd& getDeltas() const
	{
		return deltas_;
	}
	const Eigen::VectorXcd& deltaMean() const
	{
		return deltaMean_;
	}
	template<class Rhs>
	typename Machine::Vector apply(const Rhs& rhs) const
	{
		assert(rhs.size() == qs_.getDim());
		typename Machine::Vector r = deltas_*rhs;

		typename Machine::Vector res = deltas_.adjoint()*r/r.rows();
		res -= deltaMean_.conjugate()*r.mean();

		return res + Scalar(shift_)*rhs;
	}

};
} //namespace nnqs

// Implementation of nnqs::SRMatFree * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
	template<typename Rhs, typename Machine>
	struct generic_product_impl<nnqs::SRMatFree<Machine>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
	: generic_product_impl_base<nnqs::SRMatFree<Machine>, Rhs, generic_product_impl<nnqs::SRMatFree<Machine>, Rhs> >
	{
		typedef typename Product<nnqs::SRMatFree<Machine>, Rhs>::Scalar Scalar;
		template<typename Dest>
		static void scaleAndAddTo(Dest& dst, const nnqs::SRMatFree<Machine>& lhs, const Rhs& rhs, const Scalar& alpha)
		{
			// This method should implement "dst += alpha * lhs * rhs" inplace,
			// however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
			assert(alpha==Scalar(1) && "scaling is not implemented");
			EIGEN_ONLY_USED_FOR_DEBUG(alpha);
			/*
			auto r = lhs.getDeltas()*rhs;
			dst += lhs.getDeltas().adjoint()*r/r.rows();
			dst -= lhs.getDeltaMean().conjugate()*r.mean();
			dst += lhs.getShift()*rhs;
			*/
			dst += lhs.apply(rhs);
		}
	};
} //namespace internal
} //namespace Eigen

#endif//CY_SROPTIMIZER_CG_HPP
