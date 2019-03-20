#ifndef CY_STATES_RATIO_HPP
#define CY_STATES_RATIO_HPP
#include "NNQS.hpp"

template<typename T>
struct StatesData
{
	const NNQS<T>* qs1_;
	const NNQS<T>* qs2_;
	typename NNQS<T>::Vector diffA_; //a2-a1

	StatesData(const NNQS<T>* qs1, const NNQS<T>* qs2)
		: qs1_(qs1), qs2_(qs2)
	{
		update();
	}
	void update()
	{
		diffA_ = qs2_->getA();
		diffA_ -= qs1_->getA();
	}
};

template<typename T, class Derived>
class StatesRatioObj
	: public StateObj<T, Derived>
{
private:
	const NNQS<T>* qs2_;
	typename NNQS<T>::ConstVector& diffA_; // a2 - a1

public:
	StatRatioObj(const StatesData<T>* sd)
		: StatObj<T,Derived>(sd->qs1_), qs2_(sd->qs2_), diffA(sd->diffA_)
	{
	}
	

	T crossRatio() const //return qs2(sigma)/qs1(sigma)
	{
		using std::exp;
		using std::cosh;
		auto ss = static_cast<Derived*>(this)->getSigma().cast<T>();
		T ov = diffA_.transpose()*ss;
		T res = exp(ov);
#pragma omp parallel
		{
			T p_loc = 1.0;
#pragma omp for schedule(static,8)
			for(int j = 0; j < theta_.size(); j++)
			{
				p_loc *= cosh(theta2At(j))/cosh(theta1At(j));
			}
#pragma omp critical
			{
				res *= p_loc;
			}
		}
		return res;
	}

	T crossRatio(int k) const //return qs2(sigma ^ k)/qs1(sigma)
	{
		using std::exp;
		using std::cosh;
		auto ss = static_cast<Derived*>(this)->getSigma().cast<T>();
		T ov = diffA_.transpose()*ss;
		T res = exp(ov - 2.0*T(sigmaAt(k))*qs2_->A(k));
#pragma omp parallel
		{
			T p_loc = 1.0;
#pragma omp for schedule(static,8)
			for(int j = 0; j < theta_.size(); j++)
			{
				p_loc *= cosh(theta2_.coeff(j) - 2.0*T(sigma_.coeff(k))*qs2_->W(j,k))/cosh(theta1_.coeff(j));
			}
#pragma omp critical
			{
				res *= p_loc;
			}
		}
		return res;
	}
};

template<typename T>
class StatesRatioValue
	: public StatesRatioObj<StatesRatioValue<T> >
{
private:
	Eigen::VectorXd sigma_;
	typename NNQS<T>::Vector theta1_;
	typename NNQS<T>::Vector theta2_;

public:
	StatesRatioValue(const StatesData<T>* sd, Eigen::VectorXd sigma)
		: StatesRatioObj<T, StatesRatioValue<T> >(sd),
		sigma_(std::move(sigma)), 
		theta1_(sd.qs1_->calcTheta(sigma_)), theta2_(sd.qs2_->calcTheta(sigma_)),
	{
	}

	inline int sigmaAt(int i) const
	{
		return sigma_.coeff(i);
	}
	inline T theta1At(int j) const
	{
		return theta1_.coeff(j);
	}
	inline T theta2At(int j) const
	{
		return theta2_.coeff(j);
	}

	void flip(int k)
	{
		#pragma omp parallel for schedule(static,8)
		for(int j = 0; j < theta_.size(); j++)
		{
			theta1_.coeffRef(j) -= 2.0*T(sigma_.coeff(k))*(StateRatioObj<T, StateRatioValue<T> >::qs_->W(j,k));
			theta2_.coeffRef(j) -= 2.0*T(sigma_.coeff(k))*(StateRatioObj<T, StateRatioValue<T> >::qs2_->W(j,k);
		}
		sigma_.coeffRef(k) *= -1;
	}



	const Eigen::VectorXi& getSigma() const & { return sigma_; } 
	Eigen::VectorXi getSigma() && { return std::move(sigma_); } 

	typename NNQS<T>::ConstVector& getTheta1() const & { return theta1_; } 
	typename NNQS<T>::Vector getTheta1() && { return std::move(theta1_); } 

	typename NNQS<T>::ConstVector& getTheta2() const & { return theta2_; } 
	typename NNQS<T>::Vector getTheta2() && { return std::move(theta2_); } 

	std::tuple<Eigen::VectorXi, typename NNQS<T>::Vector, typename NNQS<T>::Vector> data() const
	{
		return std::make_tuple(sigma_, theta1_, theta2_);
	}
};

template<typename T, bool is_const = true>
class StatesRatioRef
	: public StateRatioObj<StateRatioValue<T> >
{
private:

	typedef typename std::conditional<is_const, const Eigen::VectorXi, Eigen::VectorXi>::type SigmaType;
	typedef typename std::conditional<is_const, typename NNQS<T>::ConstVector, typename NNQS<T>::Vector>::type ThetaType;

	SigmaType& sigma_;
	ThetaType& theta1_;
	ThetaType& theta2_;

public:
	StatesRatioRef(const StatesData<T>* sd, SigmaType& sigma, ThetaType& theta1, ThetaType& theta2)
		: StateObj<T, StatesRatioRef<T> >(sd), sigma_(sigma), theta1_(theta1), theta2_(theta2)
	{
	}

	inline int sigmaAt(int i) const
	{
		return sigma_.coeff(i);
	}
	inline T thetaAt(int j) const
	{
		return theta_.coeff(j);
	}
	Eigen::VectorXi getSigma() const { return sigma_; } 
	NNQS<T>::Vector getTheta1() const { return theta1_; } 
	NNQS<T>::Vector getTheta2() const { return theta2_; } 

};
#endif//CY_STATES_RATIO_HPP
