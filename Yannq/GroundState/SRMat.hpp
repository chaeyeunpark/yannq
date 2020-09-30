#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include <tbb/tbb.h>

#include "Utilities/Utility.hpp"
#include "Observables/Energy.hpp"
#include "./utils.hpp"

namespace yannq
{
template<typename Machine, typename Hamiltonian>
class SRMat;
} //namespace yannq

namespace Eigen { //namespace Eigen
namespace internal {
	template<typename Machine, typename Hamiltonian>
	struct traits<yannq::SRMat<Machine, Hamiltonian> > 
		:  public Eigen::internal::traits<Eigen::SparseMatrix<typename Machine::Scalar> > {};
}
}// namespace Eigen;

namespace yannq
{
//! \addtogroup GroundState

//! \ingroup GroundState
//! This class that generates the quantum Fisher matrix for the stochastic reconfiguration (SR) method.
template<typename Machine, typename Hamiltonian>
class SRMat
	: public Eigen::EigenBase<SRMat<Machine, Hamiltonian> >
{
public:
	using Scalar = typename Machine::Scalar;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Vector = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	using VectorConstRef = typename Eigen::Ref<const Vector>;
	using RealMatrix = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
	using RealVector = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

	using StorageIndex = uint32_t;

	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = false
	};

private:
	uint32_t n_;

	const Machine& qs_;
	const Hamiltonian& ham_;
	
	Energy<Scalar, Hamiltonian> energy_;

	RealScalar shift_;
	
	Matrix deltas_;
	Vector deltaMean_;

	Vector energyGrad_;
	Vector weights_;

public:
	//! \param qs Machine that describes quantum states
	//! \param ham Hamiltonian for SR
	SRMat(const Machine& qs, const Hamiltonian& ham)
	  : n_{qs.getN()}, qs_(qs), ham_(ham), energy_(ham)
	{
	}

	Eigen::Index rows() const { return qs_.getDim(); }
	Eigen::Index cols() const { return qs_.getDim(); }

	template<typename Rhs>
	Eigen::Product<SRMat<Machine, Hamiltonian>, Rhs, Eigen::AliasFreeProduct> 
			operator*(const Eigen::MatrixBase<Rhs>& x) const {
	  return Eigen::Product<SRMat<Machine, Hamiltonian>, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
	}

	//! \param rs Sampling results obtained from samplers.
	template<class SamplingResult>
	void constructFromSamples(SamplingResult&& sr)
	{
		weights_.resize(0);
		int nsmp = sr.size();

		constructDelta(qs_, sr, deltas_);
		constructObs(qs_, sr, energy_);

		deltaMean_ = deltas_.colwise().mean();
		deltas_ = deltas_.rowwise() - deltaMean_.transpose();
		
		energyGrad_ =  deltas_.adjoint() * energy_.elocs();
		energyGrad_ /= nsmp;
	}

	template<class SamplingResult>
	void constructFromWeightSamples(const Eigen::Ref<const RealVector>& weights, SamplingResult&& sr)
	{
		weights_ = weights;

		constructDelta(qs_, sr, deltas_);
		constructObsWeights(qs_, std::forward<SamplingResult>(sr), weights, energy_);

		deltaMean_ = weights.transpose()*deltas_;
		deltas_ = deltas_.rowwise() - deltaMean_.transpose();
		
		energyGrad_ =  deltas_.adjoint() * weights.asDiagonal()
			* energy_.elocs();
	}


	void setShift(RealScalar shift)
	{
		shift_ = shift;
	}

	RealScalar getShift() const
	{
		return shift_;
	}

	//! return \f$\langle \nabla_{\theta} \psi_\theta(\sigma) \rangle\f$ 
	const Vector& oloc() const&
	{
		return deltaMean_;
	}

	Vector oloc() &&
	{
		return deltaMean_;
	}

	Matrix corrMat() 
	{
		int nsmp = deltas_.rows();
		if(weights_.size() == 0)
			return (deltas_.adjoint() * deltas_)/nsmp;
		else
			return (deltas_.adjoint() * weights_.asDiagonal() * deltas_);
	}

	RealScalar eloc() const
	{
		return std::real(energy_.eloc());
	}

	Scalar elocVar() const
	{
		return std::real(energy_.elocVar());
	}
	
	const Vector& energyGrad() const&
	{
		return energyGrad_;
	}
	Vector energyGrad() &&
	{
		return energyGrad_;
	}

	Vector apply(const Vector& rhs) const
	{
		assert(rhs.size() == qs_.getDim());
		Vector r = deltas_*rhs;

		Vector res;
		
		if(weights_.size() == 0)
		{
			r /= r.rows();
			res = deltas_.adjoint()*r;
		}
		else
		{
			r.array() *= weights_.array();
			res = deltas_.adjoint() * r;
		}

		return res + Scalar(shift_)*rhs;
	}

	/*! \brief Use conjugate gradient solover to solve the optimizing vector.
	 *
	 * We solve \f$ (S + \epsilon \mathbb{1})v = f \f$ where \f$ f \f$ is the gradient of the energy expectation values. 
	 * \param shift \f$ \epsilon \f$ that controls regularization
	 * \param tol tolerance for CG solver
	 */
	Vector solveCG(RealScalar shift, RealScalar tol = 1e-4)
	{
		setShift(shift);
		Eigen::ConjugateGradient<
			SRMat<Machine, Hamiltonian>,
			Eigen::Lower|Eigen::Upper,
			Eigen::IdentityPreconditioner> cg;
		cg.compute(*this);
		cg.setTolerance(tol);
		return cg.solve(energyGrad_);
	
	}

	/**
	 * Solve S^{-1}vec using conjugate gradient method
	 */
	Vector solveCG(const VectorConstRef& vec, double shift, double tol = 1e-4)
	{
		setShift(shift);
		Eigen::ConjugateGradient<
			SRMat<Machine, Hamiltonian>,
			Eigen::Lower|Eigen::Upper,
			Eigen::IdentityPreconditioner> cg;
		cg.compute(*this);
		cg.setTolerance(tol);
		return cg.solve(vec);
	}


	/*! \brief Solve the optimizing vector by solving the linear equation exactly..
	 *
	 * We solve \f$ (S + \epsilon \mathbb{1})v = f \f$ where \f$ f \f$ is the gradient of the energy expectation values. 
	 * \param shift \f$ \epsilon \f$ that controls regularization
	 */
	Vector solveExact(RealScalar shift)
	{
		Matrix mat = corrMat();
		mat += shift*Matrix::Identity(mat.rows(),mat.cols());
		Eigen::LLT<Matrix> llt{mat};
		return llt.solve(energyGrad_);
	}

};
} //namespace yannq


namespace Eigen {
namespace internal {
	template<typename Rhs, typename Machine, typename Hamiltonian>
	struct generic_product_impl<yannq::SRMat<Machine, Hamiltonian>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
	: generic_product_impl_base<yannq::SRMat<Machine, Hamiltonian>, Rhs, generic_product_impl<yannq::SRMat<Machine, Hamiltonian>, Rhs> >
	{
		typedef typename Product<yannq::SRMat<Machine, Hamiltonian>, Rhs>::Scalar Scalar;
		template<typename Dest>
		static void scaleAndAddTo(Dest& dst, const yannq::SRMat<Machine, Hamiltonian>& lhs, const Rhs& rhs, const Scalar& alpha)
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

