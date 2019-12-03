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
/** 
 * Construct the fisher information metric for quantum states
 * */
template<typename Machine, typename Hamiltonian>
class SRMat
{
public:
	using Scalar = typename Machine::Scalar;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Vector = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
	using RealMatrix = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
	using RealVector = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
private:
	int n_;

	const Machine& qs_;
	const Hamiltonian& ham_;
	
	FisherMatrix<Machine> fisher_;
	Energy<Scalar, Hamiltonian> energy_;

	Vector energyGrad_;

public:
	SRMat(const Machine& qs, const Hamiltonian& ham)
	  : n_{qs.getN()}, qs_(qs), ham_(ham),
		fisher_(qs), energy_(ham)
	{
	}

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
	const Vector& oloc() const&
	{
		return fisher_.oloc();
	}
	Vector&& oloc() &&
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
	/**
	 * If useCG is set, we use the CG solver with given tol
	 */
	Vector solveCG(double shift, double tol = 1e-4)
	{
		fisher_.setShift(shift);
		Eigen::ConjugateGradient<FisherMatrix<Machine>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
		cg.compute(fisher_);
		cg.setTolerance(tol);
		return cg.solve(energyGrad_);
	}
	Vector solveExact(double shift)
	{
		Matrix mat = fisher_.corrMat();
		mat += shift*Matrix::Identity(mat.rows(),mat.cols());
		Eigen::LLT<Matrix> llt{mat};
		return llt.solve(energyGrad_);
	}

};
} //namespace yannq
#endif//YANNQ_GROUNDSTATES_SRMAT_HPP_HPP
