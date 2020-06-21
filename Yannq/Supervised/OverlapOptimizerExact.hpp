#ifndef YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#define YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#include <Eigen/Dense>
#include <limits>

#include "Observables/ConstructDeltaExact.hpp"
#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

#include <iterator>
#include <tbb/tbb.h>

namespace yannq
{
//! Use this if the parameter type of the machine is complex.
template<class Machine>
class OverlapOptimizerExact
{
public:
	using Scalar = typename Machine::Scalar;
	using RealScalar = typename yannq::remove_complex<Scalar>::type;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;

private:
	const int n_;
	const Machine& qs_;
	tbb::concurrent_vector<uint32_t> basis_;

	Vector target_;

	Matrix deltas_;
	Matrix psiDeltas_;
	Vector ovs_;
	Vector oloc_;
	Scalar ov_;

public:
	template<class BasisType, typename disable_if<
		std::is_same<typename std::remove_reference<BasisType>::type,
		tbb::concurrent_vector<uint32_t>>::value,
		int>::type = 0>
	explicit OverlapOptimizerExact(const Machine& qs, BasisType&& basis)
		: n_(qs.getN()), qs_(qs)
	{
		basis_ = parallelConstructBasis(basis);
	}

	explicit OverlapOptimizerExact(const Machine& qs, tbb::concurrent_vector<uint32_t>&& basis)
		: n_(qs.getN()), qs_(qs), basis_(std::move(basis))
	{
	}

	void setTarget(Vector v)
	{
		assert(basis_.size() == v.size());

		target_ = std::move(v);
	}


	const Vector& getTarget() const&
	{
		return target_;
	}

	Vector getTarget() &&
	{
		return target_;
	}

	const tbb::concurrent_vector<uint32_t>& getBasis() const&
	{
		return basis_;
	}

	tbb::concurrent_vector<uint32_t> getBasis() &&
	{
		return basis_;
	}

	void constructExact() 
	{
		using std::conj;

		Vector st = getPsi(qs_, basis_, true);

		deltas_ = constructDeltaExact(qs_, basis_);

		psiDeltas_ = st.cwiseAbs2().asDiagonal()*deltas_; 
		ovs_ = st.conjugate().cwiseProduct(target_);
		ov_ = ovs_.sum();
		oloc_ = psiDeltas_.colwise().sum();
	}
	
	//! return -\nabla_{\theta^*} \log |\langle \Phi | \psi_\theta \rangle|^2
	Vector calcLogGrad() const
	{
		Vector res = oloc().conjugate();
		Vector r1 = ovs_.transpose()*deltas_.conjugate();
		res -= r1/ov_;

		return res;
	}
	Vector calcGrad() const
	{
		Vector res = oloc().conjugate()*std::norm(ov_);
		res -= ovs_.transpose()*deltas_.conjugate()*conj(ov_);
		return res;
	}
	
	Vector oloc() const
	{
		return oloc_;
	}

	Matrix corrMat() const
	{
		Matrix res = deltas_.adjoint()*psiDeltas_;
		return res - oloc_.conjugate()*oloc_.transpose();
	}
	/**
	 * 
	 */
	Matrix uncenteredCorrMat() const
	{
		Matrix res = deltas_.adjoint()*psiDeltas_;
		//Matrix ob = ovs_.conjugate().asDiagonal()*deltas_/std::conj(ov_);
		//res -= ob.adjoint()*ob;
		return res;
	}


	/**
	 * return squared fidelity: |\langle \psi_\theta | \phi \rangle|^2
	 */
	RealScalar fidelity() const
	{
		using std::norm;
		return norm(ov_);
	}
};


} //yannq
#endif//YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
