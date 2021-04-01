#pragma once

#include "Machines/RBM.hpp"
#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"
#include "./type_traits.hpp"

namespace yannq
{

template<typename T>
class RBMStateValue;
template<typename T>
class RBMStateRef;

template<typename T>
class MachineStateTypes<RBM<T> >
{
public:
	using StateValue = RBMStateValue<T>;
	using StateRef = RBMStateRef<T>;
};

template<typename T, class Derived>
class RBMStateObj
{
protected:
	const RBM<T>& qs_;

public:
	using Scalar = T;
	using Machine = RBM<Scalar>;
	using RealScalar = typename remove_complex<Scalar>::type;

	RBMStateObj(const Machine& qs) noexcept
		: qs_(qs)
	{
	}

	/********************** logRatio for a single spin ************************/

	inline RealScalar logRatioRe(int k)
	{
		return real(logRatio(k));
	}

	/// returns log[psi(sigma ^ k)] - log[psi(sigma)]
	Scalar logRatio(int k) const 
	{
		using std::exp;
		using std::cosh;
		using std::log;
		
		Scalar res = -RealScalar(2.0)*qs_.A(k)*RealScalar(sigmaAt(k));

		int m = qs_.getM();
		for(int j = 0; j < m; j ++)
		{
			res += logCosh(thetaAt(j)-RealScalar(2.0)*RealScalar(sigmaAt(k))*qs_.W(j,k))
				-logCosh(thetaAt(j));
		}
		return res;
	}

	inline Scalar ratio(int k) const
	{
		return std::exp(logRatio(k));
	}

	/************************ logRatio for two spins **************************/

	inline RealScalar logRatioRe(int k, int l) const
	{
		return real(logRatio(k,l));
	}

	/// returns log[psi(sigma ^ k ^ l)] - log[psi(sigma)]
	Scalar logRatio(int k, int l) const 
	{
		using std::exp;
		using std::cosh;
		Scalar res = -RealScalar(2.0)*qs_.A(k)*RealScalar(sigmaAt(k))
			-RealScalar(2.0)*qs_.A(l)*RealScalar(sigmaAt(l));
		const int m = qs_.getM();

		for(int j = 0; j < m; j ++)
		{
			Scalar t = thetaAt(j)-RealScalar(2.0)*RealScalar(sigmaAt(k))*qs_.W(j,k)
				-RealScalar(2.0)*RealScalar(sigmaAt(l))*qs_.W(j,l);
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
	}

	inline Scalar ratio(int k, int l) const
	{
		return std::exp(logRatio(k,l));
	}

	/************************ logRatio for vector *****************************/

	inline RealScalar logRatioRe(const std::vector<int>& v) const
	{
		return real(logRatio(v));
	}

	/// returns log[psi(sigma ^ v)] - log[psi(sigma)]
	Scalar logRatio(const std::vector<int>& v) const
	{
		Scalar res{};
		const int m = qs_.getM();
		for(int elt: v)
		{
			res -= RealScalar(2.0)*qs_.A(elt)*RealScalar(sigmaAt(elt));
		}
		for(int j = 0; j < m; j++)
		{
			Scalar t = thetaAt(j);
			for(int elt: v)
			{
				t -= RealScalar(2.0)*RealScalar(sigmaAt(elt))*qs_.W(j,elt);
			}
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
	}

	inline Scalar ratio(const std::vector<int>& v) const
	{
		return std::exp(logRatio(v));
	}

	/********************* logRatio with another StateObj *********************/

	inline RealScalar logRatioRe(const RBMStateObj<Scalar, Derived>& other) const
	{
		return real(logRatio(other));
	}
	
	/// returns log[psi(other.sigma)] - log[psi(sigma)]
	Scalar logRatio(const RBMStateObj<Scalar, Derived>& other) const
	{
		Scalar res = (this->qs_.getA().transpose())*
			(other.getSigma() - getSigma()).template cast<Scalar>();
		const int m = qs_.getM();
		for(int j = 0; j < m; j++)
		{
			res += logCosh(other.thetaAt(j)) - logCosh(thetaAt(j));
		}
		return res;
	}

	inline const Eigen::VectorXi& getSigma() const&
	{
		return static_cast<const Derived*>(this)->getSigma();
	}

	inline int sigmaAt(int i) const
	{
		return static_cast<const Derived*>(this)->sigmaAt(i);
	}

	inline Scalar thetaAt(int i) const
	{
		return static_cast<const Derived*>(this)->thetaAt(i);
	}

	const Machine& getRBM() const
	{
		return qs_;
	}
};

template<typename T>
class RBMStateValue
	: public RBMStateObj<T, RBMStateValue<T> >
{
public:
	using Scalar = T;
	using RealScalar = typename remove_complex<Scalar>::type;
	using Vector = typename RBM<Scalar>::Vector;
	using Machine = RBM<Scalar>;

private:
	Eigen::VectorXi sigma_;
	Vector theta_;

public:

	RBMStateValue(const RBM<Scalar>& qs, Eigen::VectorXi&& sigma) noexcept
		: RBMStateObj<Scalar, RBMStateValue<Scalar> >(qs), sigma_(std::move(sigma))
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValue(const RBM<Scalar>& qs, const Eigen::VectorXi& sigma) noexcept
		: RBMStateObj<Scalar, RBMStateValue<Scalar> >(qs), sigma_(sigma)
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValue(const RBM<Scalar>& qs, const Eigen::VectorXi& sigma, const Vector& theta) noexcept
		: RBMStateObj<Scalar, RBMStateValue<Scalar> >(qs), sigma_{sigma}, theta_{theta}
	{
	}

	RBMStateValue(const RBMStateValue<Scalar>& rhs) /* noexcept */ = default;
	RBMStateValue(RBMStateValue<Scalar>&& rhs) /* noexcept */ = default;

	void swap(RBMStateValue<Scalar>& rhs) noexcept
	{
		using std::swap;
		assert(rhs.qs_ == this->qs_);
		swap(sigma_, rhs.sigma_);
		swap(theta_, rhs.theta_);
	}

	/**
	 * Update theta when the RBM changes
	 */
	void updateTheta()
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	void setSigma(const Eigen::VectorXi& sigma)
	{
		sigma_ = sigma;
		theta_ = this->qs_.calcTheta(sigma_);
	}

	void setSigma(Eigen::VectorXi&& sigma)
	{
		sigma_ = std::move(sigma);
		theta_ = this->qs_.calcTheta(sigma_);
	}

	inline int sigmaAt(int i) const
	{
		return sigma_(i);
	}
	inline Scalar thetaAt(int j) const
	{
		return theta_(j);
	}
	

	void flip(int k)
	{
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) -= RealScalar(2.0)*RealScalar(sigma_(k))*(this->qs_.W(j,k));
		}
		sigma_(k) *= -1;
	}

	void flip(int k, int l)
	{
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) += -RealScalar(2.0)*RealScalar(sigma_(k))*(this->qs_.W(j,k))
				-RealScalar(2.0)*RealScalar(sigma_(l))*(this->qs_.W(j,l));
		}
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	void flip(const std::vector<int>& v)
	{
		for(int elt: v)
		{
			for(int j = 0; j < theta_.size(); j++)
			{
				theta_(j) -= RealScalar(2.0)*RealScalar(sigma_(elt))*(this->qs_.W(j,elt));
			}
		}
		for(int elt: v)
		{
			sigma_(elt) *= -1;
		}
	}

	
	using RBMStateObj<Scalar, RBMStateValue<Scalar> >::logRatio;
	

	inline const Eigen::VectorXi& getSigma() const & { return sigma_; } 
	inline Eigen::VectorXi getSigma() && { return std::move(sigma_); } 

	inline const Vector& getTheta() const & { return theta_; } 
	inline Vector getTheta() && { return std::move(theta_); } 

	inline std::tuple<Eigen::VectorXi, Vector> data() const
	{
		return std::make_tuple(sigma_, theta_);
	}
};
template<typename T>
void swap(RBMStateValue<T>& lhs, RBMStateValue<T>& rhs)
{
	lhs.swap(rhs);
}


template<typename T>
class RBMStateRef
	: public RBMStateObj<T, RBMStateRef<T> >
{
public:
	using Scalar = T;
	using RealScalar = typename remove_complex<Scalar>::type;
	using Vector = typename RBM<Scalar>::Vector;
	using Machine = RBM<Scalar>;

private:
	const Eigen::VectorXi& sigma_;
	const Vector& theta_;

public:
	
	RBMStateRef(const RBM<Scalar>& qs, const Eigen::VectorXi& sigma, const Vector& theta) noexcept
		: RBMStateObj<Scalar, RBMStateRef<Scalar> >(qs), sigma_(sigma), theta_(theta)
	{
	}

	inline int sigmaAt(int i) const
	{
		return sigma_(i);
	}
	inline Scalar thetaAt(int j) const
	{
		return theta_(j);
	}

	inline Eigen::VectorXi getSigma() const
	{
		return sigma_;
	}

	inline const Vector& getTheta() const&
	{
		return theta_;
	}

	inline Vector getTheta() &&
	{
		return theta_;
	}

};

template<typename T>
struct is_reference_state_type<RBMStateRef<T> >: public std::true_type {};

} //namespace yannq
