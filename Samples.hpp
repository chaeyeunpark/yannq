#ifndef CY_NNQS_SAMPLES_HPP
#define CY_NNQS_SAMPLES_HPP
#include "NNQS.hpp"
#include "tuple_helper.hpp"
#include "type_traits.hpp"
#include "Utility.hpp"

namespace nnqs
{

template<typename T, class Derived>
class StateObj
{
protected:
	const NNQS<T>* qs_;
public:
	using ValueType = T;
	using AuxDataT = NNQS<T>;

	StateObj(const NNQS<T>* qs)
		: qs_(qs)
	{
	}

	T logRatio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		using std::exp;
		using std::cosh;
		using std::log;
		T res = -2.0*qs_->A(k)*T(sigmaAt(k));
		int m = qs_->getM();
		for(int j = 0; j < m; j ++)
		{
			res += logCosh(thetaAt(j)-2.0*T(sigmaAt(k))*qs_->W(j,k))
				-logCosh(thetaAt(j));
		}
		return res;
	}

	inline T ratio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		return std::exp(logRatio(k));
	}

	T logRatio(int k, int l) const //calc psi(sigma ^ k ^ l)/psi(sigma)
	{
		using std::exp;
		using std::cosh;
		T res = -2.0*qs_->A(k)*T(sigmaAt(k))-2.0*qs_->A(l)*T(sigmaAt(l));
		const int m = qs_->getM();

		for(int j = 0; j < m; j ++)
		{
			T t = thetaAt(j)-2.0*T(sigmaAt(k))*qs_->W(j,k)-2.0*T(sigmaAt(l))*qs_->W(j,l);
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
	}

	T ratio(int k, int l) const
	{
		return std::exp(logRatio(k,l));
	}
	template<std::size_t N>
	T logRatio(const std::array<int, N>& v) const
	{
		T res{};
		const int m = qs_->getM();
		for(int elt: v)
		{
			res -= 2.0*qs_->A(elt)*T(sigmaAt(elt));
		}
		for(int j = 0; j < m; j++)
		{
			T t = thetaAt(j);
			for(int elt: v)
			{
				t -= 2.0*T(sigmaAt(elt))*qs_->W(j,elt);
			}
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
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
struct StateValue
	: public StateObj<T, StateValue<T> >
{
private:
	Eigen::VectorXi sigma_;
	typename NNQS<T>::Vector theta_;

public:
	using Vector=typename NNQS<T>::Vector;

	StateValue(const NNQS<T>* qs, Eigen::VectorXi&& sigma)
		: StateObj<T, StateValue<T> >(qs), sigma_(std::move(sigma))
	{
		theta_ = StateObj<T, StateValue<T> >::qs_->calcTheta(sigma_);
	}

	StateValue(const NNQS<T>* qs, const Eigen::VectorXi& sigma)
		: StateObj<T, StateValue<T> >(qs), sigma_(sigma)
	{
		theta_ = StateObj<T, StateValue<T> >::qs_->calcTheta(sigma_);
	}

	StateValue(const StateValue<T>& rhs) = default;
	StateValue(StateValue<T>&& rhs) = default;

	StateValue& operator=(StateValue<T>&& rhs)
	{
		StateObj<T,StateValue<T> >::operator=(rhs);
		sigma_ = std::move(rhs.sigma_);
		theta_ = std::move(rhs.theta_);
		return *this;
	}

	void setSigma(const Eigen::VectorXi& sigma)
	{
		sigma_ = sigma;
		theta_ = StateObj<T, StateValue<T> >::qs_->calcTheta(sigma_);
	}

	void setSigma(Eigen::VectorXi&& sigma)
	{
		sigma_ = std::move(sigma);
		theta_ = StateObj<T, StateValue<T> >::qs_->calcTheta(sigma_);
	}

	inline int sigmaAt(int i) const
	{
		return sigma_(i);
	}
	inline T thetaAt(int j) const
	{
		return theta_(j);
	}
	
	template<std::size_t N>
	void flip(const std::array<int, N>& v)
	{
		for(int elt: v)
		{
			for(int j = 0; j < theta_.size(); j++)
			{
				theta_(j) -= 2.0*T(sigma_(elt))*(StateObj<T, StateValue<T> >::qs_->W(j,elt));
			}
		}
		for(int elt: v)
		{
			sigma_(elt) *= -1;
		}
	}

	void flip(int k, int l)
	{
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) += -2.0*T(sigma_(k))*(StateObj<T, StateValue<T> >::qs_->W(j,k))
				-2.0*T(sigma_(l))*(StateObj<T, StateValue<T> >::qs_->W(j,l));
		}
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	void flip(int k)
	{
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) -= 2.0*T(sigma_(k))*(StateObj<T, StateValue<T> >::qs_->W(j,k));
		}
		sigma_(k) *= -1;
	}
	
	/*
	T flipAndLogRatio(Eigen::VectorXi to)
	{
		using std::exp;
		using std::cosh;
		using std::log;

		Vector theta_p = StateObj<T, StateValue<T> >::qs_->calcTheta(to);
		to -= sigma_;
		T res = (StateObj<T, StateValue<T> >::qs_->getA().transpose())*to.cast<T>();

		for(int j = 0; j < theta_.size(); j++)
		{
			res += logCosh(theta_p(j)) - logCosh(theta_(j));
		}
		sigma_ += to;
		return res;
	}
	*/
	
	using StateObj<T, StateValue<T> >::logRatio;
	T logRatio(const StateValue& other)
	{
		T res = (StateObj<T, StateValue<T> >::qs_->getA().transpose())*
			(other.getSigma() - sigma_).template cast<T>();
		for(int j = 0; j < theta_.size(); j++)
		{
			res += logCosh(other.theta_(j)) - logCosh(theta_(j));
		}
		return res;
	}

	const Eigen::VectorXi& getSigma() const & { return sigma_; } 
	Eigen::VectorXi getSigma() && { return std::move(sigma_); } 

	typename NNQS<T>::ConstVector& getTheta() const & { return theta_; } 
	typename NNQS<T>::Vector getTheta() && { return std::move(theta_); } 

	std::tuple<Eigen::VectorXi, typename NNQS<T>::Vector> data() const
	{
		return std::make_tuple(sigma_, theta_);
	}
};


template<typename T, bool is_const = true>
class StateRef
	: public StateObj<T, StateRef<T> >
{
private:
	typedef typename std::conditional<is_const, const Eigen::VectorXi, Eigen::VectorXi>::type SigmaType;
	typedef typename std::conditional<is_const, typename NNQS<T>::ConstVector, typename NNQS<T>::Vector>::type ThetaType;
	SigmaType& sigma_;
	ThetaType& theta_;
public:
	StateRef(const NNQS<T>* qs, SigmaType& sigma, ThetaType& theta)
		: StateObj<T, StateRef<T> >(qs), sigma_(sigma), theta_(theta)
	{
	}

	inline int sigmaAt(int i) const
	{
		return sigma_(i);
	}
	inline T thetaAt(int j) const
	{
		return theta_(j);
	}

	Eigen::VectorXi getSigma() const
	{
		return sigma_;
	}

	typename NNQS<T>::Vector getTheta() const
	{
		return theta_;
	}

};

template<typename T>
struct is_reference_state_type<StateRef<T> >: public std::true_type {};

} //namespace nnqs
#endif//CY_NNQS_SAMPLES_HPP
