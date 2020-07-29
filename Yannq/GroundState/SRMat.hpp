#ifndef YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
#define YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include <tbb/tbb.h>

#include "Utilities/Utility.hpp"
#include "Observables/FisherMatrix.hpp"
#include "Observables/Energy.hpp"

namespace yannq
{
//! \addtogroup GroundState

//! \ingroup GroundState
//! This class that generates the quantum Fisher matrix for the stochastic reconfiguration (SR) method.
template<typename Machine, typename Hamiltonian>
class SRMat
{
public:
	using Scalar = typename Machine::Scalar;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Vector = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	using VectorConstRef = typename Eigen::Ref<const Vector>;
	using RealMatrix = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
	using RealVector = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
private:
	uint32_t n_;

	const Machine& qs_;
	const Hamiltonian& ham_;
	
	FisherMatrix<Machine> fisher_;
	Energy<Scalar, Hamiltonian> energy_;

	Vector energyGrad_;

public:
	//! \param qs Machine that describes quantum states
	//! \param ham Hamiltonian for SR
	SRMat(const Machine& qs, const Hamiltonian& ham)
	  : n_{qs.getN()}, qs_(qs), ham_(ham),
		fisher_(qs), energy_(ham)
	{
	}

	//! \param rs Sampling results obtained from samplers.
	template<class SamplingResult>
	void constructFromSampling(SamplingResult&& rs)
	{
		int nsmp = rs.size();
		fisher_.initIter(nsmp);
		energy_.initIter(nsmp);

		tbb::parallel_for(std::size_t(0u), rs.size(),
			[&](uint32_t idx)
		{
			const auto& elt = rs[idx];
			auto state = construct_state<typename MachineStateTypes<Machine>::StateRef>(qs_, elt);
			fisher_.eachSample(idx, elt, state);
			energy_.eachSample(idx, elt, state);
		});

		fisher_.finIter();
		energy_.finIter();
		
		auto derv = fisher_.logDervs().adjoint();

		energyGrad_ =  derv * energy_.elocs();
		energyGrad_ /= nsmp;
	}

	//! return \f$\langle \nabla_{\theta} \psi_\theta(\sigma) \rangle\f$ 
	const Vector& oloc() const&
	{
		return fisher_.oloc();
	}
	Vector oloc() &&
	{
		return fisher_.oloc();
	}

	Matrix corrMat() 
	{
		return fisher_.corrMat();
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

	Vector apply(const Vector& v) const
	{
		return fisher_.apply(v);
	}

	/*! \brief Use conjugate gradient solover to solve the optimizing vector.
	 *
	 * We solve \f$ (S + \epsilon \mathbb{1})v = f \f$ where \f$ f \f$ is the gradient of the energy expectation values. 
	 * \param shift \f$ \epsilon \f$ that controls regularization
	 * \param tol tolerance for CG solver
	 */
	Vector solveCG(RealScalar shift, RealScalar tol = 1e-4)
	{
		fisher_.setShift(shift);
		Eigen::ConjugateGradient<
			FisherMatrix<Machine>,
			Eigen::Lower|Eigen::Upper,
			Eigen::IdentityPreconditioner> cg;
		cg.compute(fisher_);
		cg.setTolerance(tol);
		return cg.solve(energyGrad_);
	
	}

	/**
	 * Solve S^{-1}vec using conjugate gradient method
	 */
	Vector solveCG(const VectorConstRef& vec, double shift, double tol = 1e-4)
	{
		fisher_.setShift(shift);
		Eigen::ConjugateGradient<
			FisherMatrix<Machine>,
			Eigen::Lower|Eigen::Upper,
			Eigen::IdentityPreconditioner> cg;
		cg.compute(fisher_);
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
		Matrix mat = fisher_.corrMat();
		mat += shift*Matrix::Identity(mat.rows(),mat.cols());
		Eigen::LLT<Matrix> llt{mat};
		return llt.solve(energyGrad_);
	}

};
} //namespace yannq
#endif//YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
