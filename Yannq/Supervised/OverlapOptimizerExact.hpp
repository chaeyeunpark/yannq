#ifndef YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#define YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#include <Eigen/Dense>
#include <limits>
#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

namespace yannq
{

template<class Machine, class Enable = void>
class OverlapOptimizerExact {};

//! Use this if the parameter type of the machine is complex.
template<class Machine>
class OverlapOptimizerExact<Machine, std::enable_if_t<is_complex_type<typename Machine::ScalarType>::value> >
{
public:
	using ScalarType = typename Machine::ScalarType;
	using RealScalarType = typename yannq::remove_complex<ScalarType>::type;
	using VectorType = typename Machine::VectorType;
	using MatrixType = typename Machine::MatrixType;

private:
	const Machine& qs_;
	const int N_;
	VectorType target_;

	MatrixType deltas_;
	MatrixType psiDeltas_;
	VectorType ovs_;
	VectorType oloc_;
	ScalarType ov_;

public:
	template<class Target>
	explicit OverlapOptimizerExact(const Machine& qs, const Target& t)
		: qs_(qs), N_(qs.getN())
	{
		target_.resize(1<<N_);
		for(int i = 0; i < (1<<N_); i++)
		{
			target_(i) = t(i);
		}
	}

	const VectorType& getTarget() const&
	{
		return target_;
	}

	VectorType getTarget() &&
	{
		return target_;
	}

	void constructExact() 
	{
		using std::conj;

		deltas_.setZero(1<<N_, qs_.getDim());

		VectorType st = getPsi(qs_, true);

#pragma omp parallel for schedule(static,8)
		for(uint32_t n = 0; n < (1u<<N_); n++)
		{
			auto s = toSigma(N_, n);
			deltas_.row(n) = qs_.logDeriv(qs_.makeData(s));
		}
		psiDeltas_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		ovs_ = st.conjugate().cwiseProduct(target_);
		ov_ = ovs_.sum();
		oloc_ = psiDeltas_.colwise().sum();
	}
	
	//! return -\nabla_{\theta^*} \log |\langle \Phi | \psi_\theta \rangle|^2
	VectorType calcLogGrad() const
	{
		VectorType res = oloc().conjugate();
		VectorType r1 = ovs_.transpose()*deltas_.conjugate();
		res -= r1/ov_;

		return res;
	}
	VectorType calcGrad() const
	{
		VectorType res = oloc().conjugate()*std::norm(ov_);
		res -= ovs_.transpose()*deltas_.conjugate()*conj(ov_);
		return res;
	}
	
	VectorType oloc() const
	{
		return oloc_;
	}

	MatrixType corrMat() const
	{
		MatrixType res = deltas_.adjoint()*psiDeltas_;
		return res - oloc_.conjugate()*oloc_.transpose();
	}

	/**
	 * return squared fidelity: |\langle \psi_\theta | \phi \rangle|^2
	 */
	double fidelity() const
	{
		using std::norm;
		return norm(ov_);
	}
};

//! Use this if the parameter type of the machine is real.
template<class Machine>
class OverlapOptimizerExact<Machine, std::enable_if_t<!is_complex_type<typename Machine::ScalarType>::value> >
{
public:
	using ReScalarType = typename Machine::ScalarType;
	using CxScalarType = std::complex<typename Machine::ScalarType>;
	using ReVectorType = Eigen::Matrix<ReScalarType, Eigen::Dynamic, 1>;
	using CxVectorType = Eigen::Matrix<CxScalarType, Eigen::Dynamic, 1>;
	using ReMatrixType = Eigen::Matrix<ReScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using CxMatrixType = Eigen::Matrix<CxScalarType, Eigen::Dynamic, Eigen::Dynamic>;

private:
	const Machine& qs_;
	const int N_;
	CxVectorType target_;

	CxMatrixType deltas_;
	CxMatrixType psiDeltas_;
	CxVectorType ovs_;
	CxVectorType oloc_;
	CxScalarType ov_;

public:
	template<class Target>
	explicit OverlapOptimizerExact(const Machine& qs, const Target& t)
		: qs_(qs), N_(qs.getN())
	{
		target_.resize(1<<N_);
		for(int i = 0; i < (1<<N_); i++)
		{
			target_(i) = t(i);
		}
	}

	const CxVectorType& getTarget() const&
	{
		return target_;
	}

	CxVectorType getTarget() &&
	{
		return target_;
	}

	void constructExact() 
	{
		using std::conj;

		deltas_.setZero(1<<N_, qs_.getDim());

		CxVectorType st = getPsi(qs_, true);

#pragma omp parallel for schedule(static,8)
		for(uint32_t n = 0; n < (1u<<N_); n++)
		{
			auto s = toSigma(N_, n);
			deltas_.row(n) = qs_.logDeriv(qs_.makeData(s));
		}
		psiDeltas_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		ovs_ = st.conjugate().cwiseProduct(target_);
		ov_ = ovs_.sum();
		oloc_ = psiDeltas_.colwise().sum();
	}
	
	//! return -\nabla_{\theta} \log |\langle \Phi | \psi_\theta \rangle|^2
	ReVectorType calcLogGrad() const
	{
		ReVectorType res = oloc_.real();
		res -= (ovs_.transpose()*deltas_.conjugate()/ov_).real();

		return res;
	}
	ReVectorType calcGrad() const
	{
		ReVectorType res = oloc_.real()*std::norm(ov_);
		res -= (ovs_.transpose()*deltas_.conjugate()*conj(ov_)).real();
		return res;
	}
	
	CxVectorType oloc() const
	{
		return oloc_;
	}

	CxMatrixType corrMat() const
	{
		CxMatrixType res = deltas_.adjoint()*psiDeltas_;
		return res - oloc_.conjugate()*oloc_.transpose();
	}

	/**
	 * return squared fidelity: |\langle \psi_\theta | \phi \rangle|^2
	 */
	double fidelity() const
	{
		using std::norm;
		return norm(ov_);
	}
};

} //yannq
#endif//YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
