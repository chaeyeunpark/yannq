#ifndef YANNQ_GEOMETRY_FISHERMATIRX_HPP
#define YANNQ_GEOMETRY_FISHERMATIRX_HPP
#include <Eigen/Core>
#include <Eigen/Dense>

#include "Utilities/Utility.hpp"
#include "Observables/Observable.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace yannq
{
template<typename Machine>
class FisherMatrix;
} //namespace yannq

namespace Eigen {
namespace internal {
	template<typename Machine>
	struct traits<yannq::FisherMatrix<Machine> > :  public Eigen::internal::traits<Eigen::SparseMatrix<typename Machine::ScalarType> > {};
}
} //namespace Eigen

namespace yannq
{
/** 
 * Construct the fisher information metric for quantum states
 * */
template<class Machine>
class FisherMatrix
	: public Observable<FisherMatrix<Machine> >, 
	public Eigen::EigenBase<FisherMatrix<Machine> >
{
public:
	using Scalar = typename Machine::ScalarType;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Vector = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
	using RealMatrix = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
	using RealVector = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
private:
	int n_;

	const Machine& qs_;
	RealScalar shift_;
	
	Matrix deltas_;
	Vector deltaMean_;

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
	Eigen::Product<FisherMatrix<Machine>, Rhs, Eigen::AliasFreeProduct> 
			operator*(const Eigen::MatrixBase<Rhs>& x) const {
	  return Eigen::Product<FisherMatrix<Machine>, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
	}

	void setShift(RealScalar shift)
	{
		shift_ = shift;
	}

	RealScalar getShift() const
	{
		return shift_;
	}

	void initIter(int nsmp)
	{
		deltas_.setZero(nsmp, qs_.getDim());
	}

	template<class Elt, class State>
	inline void eachSample(int n, Elt&& elt, State&& state)
	{
		(void)state;
		deltas_.row(n) = qs_.logDeriv(elt);
	}

	void finIter()
	{
		deltaMean_ = deltas_.colwise().mean();
		deltas_ = deltas_.rowwise() - deltaMean_.transpose();
	}

	inline 
	const Matrix& logDervs() const&
	{
		return deltas_;
	}
	inline 
	Matrix logDervs() &&
	{
		return std::move(deltas_);
	}

	inline 
	const Vector& oloc() const&
	{
		return deltaMean_;
	}
	inline 
	Vector oloc() &&
	{
		return std::move(deltaMean_);
	}

	Matrix corrMat() const
	{
		int nsmp = deltas_.rows();
		return (deltas_.adjoint() * deltas_)/nsmp;
	}

	RealVector diagCorrMat() const
	{
		RealMatrix sqrDeltas = deltas_.cwiseAbs2();
		return sqrDeltas.colwise().mean();
	}
	
	FisherMatrix(const Machine& qs)
	  : n_{qs.getN()}, qs_(qs), shift_{1e-3}
	{
	}

	template<class Rhs>
	typename Machine::VectorType apply(const Rhs& rhs) const
	{
		assert(rhs.size() == qs_.getDim());
		typename Machine::VectorType r = deltas_*rhs;

		typename Machine::VectorType res = deltas_.adjoint()*r/r.rows();

		return res + Scalar(shift_)*rhs;
	}
};
}//namespace yannq

// Implementation of yannq::SRMat * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
	template<typename Rhs, typename Machine>
	struct generic_product_impl<yannq::FisherMatrix<Machine>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
	: generic_product_impl_base<yannq::FisherMatrix<Machine>, Rhs, generic_product_impl<yannq::FisherMatrix<Machine>, Rhs> >
	{
		typedef typename Product<yannq::FisherMatrix<Machine>, Rhs>::Scalar Scalar;
		template<typename Dest>
		static void scaleAndAddTo(Dest& dst, const yannq::FisherMatrix<Machine>& lhs, const Rhs& rhs, const Scalar& alpha)
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

#endif//YANNQ_GEOMETRY_FISHERMATIRX_HPP
