#ifndef YANNQ_OBSERVABLES_ENERGY_HPP
#define YANNQ_OBSERVABLES_ENERGY_HPP
#include <Eigen/Dense>
#include "Observables/Observable.hpp"

namespace yannq
{
template<typename Scalar, class Hamiltonian>
class Energy
	: public Observable<Energy<Scalar, Hamiltonian> >
{
public:
	using Vector = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
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

	void initIter(int nsmp)
	{
		elocs_.setZero(nsmp);
	}
	template<class Elt, class State>
	void eachSample(int n, Elt&& elt, State&& state)
	{
		elocs_(n) = ham_(state);
	}
	void finIter()
	{
		eloc_ = real(elocs_.mean());
		elocVar_ = elocs_.real().cwiseAbs2().sum()/elocs_.rows() - eloc_*eloc_;
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
