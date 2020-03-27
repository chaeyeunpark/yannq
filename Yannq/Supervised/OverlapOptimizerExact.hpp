#ifndef YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#define YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#include <Eigen/Dense>
#include <limits>

#include "Observables/ConstructDeltaExact.hpp"
#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

namespace yannq
{


//! Use this if the parameter type of the machine is complex.
template<class Machine>
class OverlapOptimizerExact
{
public:
	using ScalarType = typename Machine::ScalarType;
	using RealScalarType = typename yannq::remove_complex<ScalarType>::type;
	using VectorType = typename Machine::VectorType;
	using MatrixType = typename Machine::MatrixType;

private:
	const int n_;
	const Machine& qs_;
	std::vector<uint32_t> basis_;

	VectorType target_;

	MatrixType deltas_;
	MatrixType psiDeltas_;
	VectorType ovs_;
	VectorType oloc_;
	ScalarType ov_;

public:
	template<class Iterable>
	explicit OverlapOptimizerExact(const Machine& qs, Iterable&& basis)
		: n_(qs.getN()), qs_(qs)
	{
		for(auto elt: basis)
		{
			basis_.emplace_back(elt);
		}
	}

	void setTarget(VectorType v)
	{
		assert(basis_.size() == v.size());

		target_ = std::move(v);
	}


	const VectorType& getTarget() const&
	{
		return target_;
	}

	VectorType getTarget() &&
	{
		return target_;
	}

	const std::vector<uint32_t>& getBasis() const&
	{
		return basis_;
	}

	std::vector<uint32_t> getBasis() &&
	{
		return basis_;
	}

	void constructExact() 
	{
		using std::conj;

		VectorType st = getPsi(qs_, basis_, true);

		deltas_ = constructDeltaExact(qs_, basis_);

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
	 * 
	 */
	MatrixType uncenteredCorrMat() const
	{
		MatrixType res = deltas_.adjoint()*psiDeltas_;
		//MatrixType ob = ovs_.conjugate().asDiagonal()*deltas_/std::conj(ov_);
		//res -= ob.adjoint()*ob;
		return res;
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
