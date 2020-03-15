#ifndef YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
#define YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/IterativeLinearSolvers>

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
	using ScalarType = typename Machine::ScalarType;
	using RealScalarType = typename remove_complex<ScalarType>::type;

	using MatrixType = typename Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorType = typename Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
	using RealMatrixType = typename Eigen::Matrix<RealScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using RealVectorType = typename Eigen::Matrix<RealScalarType, Eigen::Dynamic, 1>;
private:
	int n_;

	const Machine& qs_;
	const Hamiltonian& ham_;
	
	FisherMatrix<Machine> fisher_;
	Energy<ScalarType, Hamiltonian> energy_;

	VectorType energyGrad_;

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

#pragma omp parallel for schedule(static,8)
		for(std::size_t n = 0; n < rs.size(); n++)
		{
			const auto& elt = rs[n];
			auto state = construct_state<typename MachineStateTypes<Machine>::StateRef>(qs_, elt);
			fisher_.eachSample(n, elt, state);
			energy_.eachSample(n, elt, state);
		}

		fisher_.finIter();
		energy_.finIter();
		
		auto derv = fisher_.logDervs().adjoint();

		energyGrad_ =  derv * energy_.elocs();
		energyGrad_ /= nsmp;
	}

	//! return \f$\langle \nabla_{\theta} \psi_\theta(\sigma) \rangle\f$ 
	const VectorType& oloc() const&
	{
		return fisher_.oloc();
	}
	VectorType&& oloc() &&
	{
		return fisher_.oloc();
	}

	MatrixType corrMat() 
	{
		return fisher_.corrMat();
	}

	RealScalarType eloc() const
	{
		return std::real(energy_.eloc());
	}

	ScalarType elocVar() const
	{
		return std::real(energy_.elocVar());
	}
	
	const VectorType& energyGrad() const&
	{
		return energyGrad_;
	}
	VectorType energyGrad() &&
	{
		return energyGrad_;
	}

	VectorType apply(const VectorType& v) const
	{
		return fisher_.apply(v);
	}

	/*! \brief Use conjugate gradient solover to solve the optimizing vector.
	 *
	 * We solve \f$ (S + \epsilon \mathbb{1})v = f \f$ where \f$ f \f$ is the gradient of the energy expectation values. 
	 * \param shift \f$ \epsilon \f$ that controls regularization
	 * \param tol tolerance for CG solver
	 */
	VectorType solveCG(double shift, double tol = 1e-4)
	{
		fisher_.setShift(shift);
		Eigen::ConjugateGradient<FisherMatrix<Machine>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
		cg.compute(fisher_);
		cg.setTolerance(tol);
		return cg.solve(energyGrad_);
	}

	VectorType solveExact(double shift)
	{
		MatrixType mat = fisher_.corrMat();
		mat += shift*MatrixType::Identity(mat.rows(),mat.cols());
		Eigen::LLT<MatrixType> llt{mat};
		return llt.solve(energyGrad_);
	}

};
} //namespace yannq
#endif//YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
