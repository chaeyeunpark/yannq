#ifndef CY_NNQS_SAMPLES_MT_HPP
#define CY_NNQS_SAMPLES_MT_HPP
#include "NNQS.hpp"
#include "tuple_helper.hpp"
#include "type_traits.hpp"
#include "Utility.hpp"

namespace nnqs
{

template<typename T, class Derived>
class StateObjMT
{
protected:
	const NNQS<T>* qs_;
public:
	using ValueType = T;
	using AuxDataT = NNQS<T>;

	StateObjMT(const NNQS<T>* qs)
		: qs_(qs)
	{
	}

	T logRatio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		using std::exp;
		using std::cosh;
		using std::log;
		int m = qs_->getM();
		double re = 0.0;
		double im = 0.0;
#pragma omp parallel for schedule(static,4) reduction(+:re,im)
		for(int j = 0; j < m; j ++)
		{
			T v = logCosh(thetaAt(j)-2.0*T(sigmaAt(k))*qs_->W(j,k))
				-logCosh(thetaAt(j));
			re += real(v);
			im += imag(v);
		}
		return -2.0*qs_->A(k)*T(sigmaAt(k)) + T{re, im};
	}

	inline T ratio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		return std::exp(logRatio(k));
	}

	T logRatio(int k, int l) const //calc psi(sigma ^ k ^ l)/psi(sigma)
	{
		using std::exp;
		using std::cosh;
		const int m = qs_->getM();
		double re = 0.0;
		double im = 0.0;
#pragma omp parallel for schedule(static,4) reduction(+:re,im)
		for(int j = 0; j < m; j ++)
		{
			T t = thetaAt(j)-2.0*T(sigmaAt(k))*qs_->W(j,k)-2.0*T(sigmaAt(l))*qs_->W(j,l);
			T v = logCosh(t)-logCosh(thetaAt(j));
			re += real(v);
			im += imag(v);
		}
		return -2.0*qs_->A(k)*T(sigmaAt(k))-2.0*qs_->A(l)*T(sigmaAt(l)) + T{re, im};
	}

	T ratio(int k, int l) const
	{
		return std::exp(logRatio(k,l));
	}

	inline int sigmaAt(int i) const
	{
		return static_cast<const Derived*>(this)->sigmaAt(i);
	}

	inline T thetaAt(int i) const
	{
		return static_cast<const Derived*>(this)->thetaAt(i);
	}

	const NNQS<T>* getNNQS() const
	{
		return qs_;
	}

	T logRatio(Eigen::VectorXi to) const
	{
		typename NNQS<T>::Vector thetaTo = qs_->calcTheta(to);
		to -= static_cast<const Derived*>(this)->getSigma();
		T s = (qs_->getA().transpose())*(to.cast<T>());
		for(int j = 0; j < qs_->getM(); j++)
		{
			s += logCosh(thetaTo(j)) - logCosh(thetaAt(j));
		}
		return s;
	}

};

template<typename T>
struct StateValueMT
	: public StateObjMT<T, StateValueMT<T> >
{
private:
	Eigen::VectorXi sigma_;
	typename NNQS<T>::Vector theta_;

public:
	using Vector=typename NNQS<T>::Vector;
	StateValueMT(const NNQS<T>* qs, Eigen::VectorXi&& sigma)
		: StateObjMT<T, StateValueMT<T> >(qs), sigma_(std::move(sigma))
	{
		theta_ = StateObjMT<T, StateValueMT<T> >::qs_->calcTheta(sigma_);
	}
	StateValueMT(const NNQS<T>* qs, const Eigen::VectorXi& sigma)
		: StateObjMT<T, StateValueMT<T> >(qs), sigma_(sigma)
	{
		theta_ = StateObjMT<T, StateValueMT<T> >::qs_->calcTheta(sigma_);
	}

	StateValueMT(const StateValueMT<T>& rhs) = default;
	StateValueMT(StateValueMT<T>&& rhs) = default;

	StateValueMT& operator=(StateValueMT<T>&& rhs)
	{
		StateObjMT<T,StateValueMT<T> >::operator=(rhs);
		sigma_ = std::move(rhs.sigma_);
		theta_ = std::move(rhs.theta_);
		return *this;
	}

	void setSigma(const Eigen::VectorXi& sigma)
	{
		sigma_ = sigma;
		theta_ = StateObjMT<T, StateValueMT<T> >::qs_->calcTheta(sigma_);
	}
	void setSigma(Eigen::VectorXi&& sigma)
	{
		sigma_ = std::move(sigma);
		theta_ = StateObjMT<T, StateValueMT<T> >::qs_->calcTheta(sigma_);
	}

	inline int sigmaAt(int i) const
	{
		return sigma_(i);
	}
	inline T thetaAt(int j) const
	{
		return theta_(j);
	}


	void flip(int k, int l)
	{
#pragma omp parallel for schedule(static,4)
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) += -2.0*T(sigma_(k))*(StateObjMT<T, StateValueMT<T> >::qs_->W(j,k))
				-2.0*T(sigma_(l))*(StateObjMT<T, StateValueMT<T> >::qs_->W(j,l));
		}
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	void flip(int k)
	{
#pragma omp parallel for schedule(static,4)
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) -= 2.0*T(sigma_(k))*(StateObjMT<T, StateValueMT<T> >::qs_->W(j,k));
		}
		sigma_(k) *= -1;
	}
	
		
	using StateObjMT<T, StateValueMT<T> >::logRatio;

	const Eigen::VectorXi& getSigma() const & { return sigma_; } 
	Eigen::VectorXi getSigma() && { return std::move(sigma_); } 

	typename NNQS<T>::ConstVector& getTheta() const & { return theta_; } 
	typename NNQS<T>::Vector getTheta() && { return std::move(theta_); } 

	std::tuple<Eigen::VectorXi, typename NNQS<T>::Vector> data() const
	{
		return std::make_tuple(sigma_, theta_);
	}
};

} //namespace nnqs
#endif//CY_NNQS_SAMPLES_MT_HPP
