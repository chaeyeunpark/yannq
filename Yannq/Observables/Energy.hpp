#ifndef YANNQ_OBSERVABLES_ENERGY_HPP
#define YANNQ_OBSERVABLES_ENERGY_HPP
#include <Eigen/Dense>
#include "Observables/Observable.hpp"
#include "Utilities/Utility.hpp"

namespace yannq
{
template<typename T, class Hamiltonian>
class Energy
	: public Observable<Energy<T, Hamiltonian> >
{
public:
	using Scalar = T;
	using RealScalar = typename remove_complex<Scalar>::type;

	using Matrix = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Vector = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
	using RealMatrix = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
	using RealVector = typename Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

private:
	const Hamiltonian& ham_;
	
	Vector elocs_;
	Scalar eloc_;
	Scalar elocVar_;

public:
	Energy(const Hamiltonian& ham)
		: ham_(ham)
	{
	}

	template<class State>
	double operator()(State&& st) const
	{
		return ham_(st);
	}

	void initIter(uint32_t nsmp)
	{
		elocs_.setZero(nsmp);
		elocVar_ = 0.;
	}

	template<class State>
	inline void eachSample(uint32_t n, State&& state)
	{
		elocs_(n) = ham_(state);
	}

	void finIter()
	{
		eloc_ = real(elocs_.mean());
		elocVar_ = elocs_.real().cwiseAbs2().mean();
		elocVar_ -= eloc_*eloc_;
	}

	template<typename MatrixT>
	void finIter(const Eigen::MatrixBase<MatrixT>& weights)
	{
		assert(weights.cols() == 1);
		eloc_ = weights.transpose()*elocs_.real();
		elocVar_ = weights.transpose()*elocs_.real().cwiseAbs2();
		elocVar_ -= eloc_*eloc_;
	}

	const Vector& elocs() const &
	{
		return elocs_;
	}

	Vector&& elocs() &&
	{
		return std::move(elocs_);
	}

	Scalar eloc() const
	{
		return eloc_;
	}

	Scalar elocVar() const
	{
		return elocVar_;
	}
};

}

#endif//YANNQ_OBSERVABLES_ENERGY_HPP
