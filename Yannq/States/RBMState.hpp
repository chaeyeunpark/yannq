#ifndef YANNQ_STATES_RBMSTATE_HPP
#define YANNQ_STATES_RBMSTATE_HPP

#include "Machines/RBM.hpp"
#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"

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

template<typename ScalarType, class Derived>
class RBMStateObj
{
protected:
	const RBM<ScalarType>& qs_;
public:
	using Machine = RBM<ScalarType>;
	using RealScalarType = typename remove_complex<ScalarType>::type;

	RBMStateObj(const Machine& qs) noexcept
		: qs_(qs)
	{
	}

	/********************** logRatio for a single spin ************************/

	inline RealScalarType logRatioRe(int k)
	{
		return real(logRatio(k));
	}

	/// returns log[psi(sigma ^ k)] - log[psi(sigma)]
	ScalarType logRatio(int k) const 
	{
		using std::exp;
		using std::cosh;
		using std::log;
		
		ScalarType res = -2.0*qs_.A(k)*RealScalarType(sigmaAt(k));

		int m = qs_.getM();
		for(int j = 0; j < m; j ++)
		{
			res += logCosh(thetaAt(j)-2.0*RealScalarType(sigmaAt(k))*qs_.W(j,k))
				-logCosh(thetaAt(j));
		}
		return res;
	}

	inline ScalarType ratio(int k) const
	{
		return std::exp(logRatio(k));
	}

	/************************ logRatio for two spins **************************/

	inline RealScalarType logRatioRe(int k, int l)
	{
		return real(logRatio(k,l));
	}

	/// returns log[psi(sigma ^ k ^ l)] - log[psi(sigma)]
	ScalarType logRatio(int k, int l) const 
	{
		using std::exp;
		using std::cosh;
		ScalarType res = -2.0*qs_.A(k)*RealScalarType(sigmaAt(k))
			-2.0*qs_.A(l)*RealScalarType(sigmaAt(l));
		const int m = qs_.getM();

		for(int j = 0; j < m; j ++)
		{
			ScalarType t = thetaAt(j)-2.0*RealScalarType(sigmaAt(k))*qs_.W(j,k)
				-2.0*RealScalarType(sigmaAt(l))*qs_.W(j,l);
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
	}

	ScalarType ratio(int k, int l) const
	{
		return std::exp(logRatio(k,l));
	}

	/************************ logRatio for vector *****************************/

	inline RealScalarType logRatioRe(const std::vector<int>& v)
	{
		return real(logRatio(v));
	}

	/// returns log[psi(sigma ^ v)] - log[psi(sigma)]
	ScalarType logRatio(const std::vector<int>& v) const
	{
		ScalarType res{};
		const int m = qs_.getM();
		for(int elt: v)
		{
			res -= 2.0*qs_.A(elt)*RealScalarType(sigmaAt(elt));
		}
		for(int j = 0; j < m; j++)
		{
			ScalarType t = thetaAt(j);
			for(int elt: v)
			{
				t -= 2.0*RealScalarType(sigmaAt(elt))*qs_.W(j,elt);
			}
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
	}

	/********************* logRatio with another StateObj *********************/

	inline RealScalarType logRatioRe(const RBMStateObj<ScalarType, Derived>& other)
	{
		return real(logRatio(other));
	}
	
	/// returns log[psi(other.sigma)] - log[psi(sigma)]
	ScalarType logRatio(const RBMStateObj<ScalarType, Derived>& other)
	{
		ScalarType res = (this->qs_.getA().transpose())*
			(other.getSigma() - getSigma()).template cast<ScalarType>();
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

	inline ScalarType thetaAt(int i) const
	{
		return static_cast<const Derived*>(this)->thetaAt(i);
	}

	const Machine& getRBM() const
	{
		return qs_;
	}
};

template<typename ScalarType>
class RBMStateValue
	: public RBMStateObj<ScalarType, RBMStateValue<ScalarType> >
{
public:
	using RealScalarType = typename remove_complex<ScalarType>::type;
	using VectorType = typename RBM<ScalarType>::VectorType;

private:
	Eigen::VectorXi sigma_;
	VectorType theta_;

public:

	RBMStateValue(const RBM<ScalarType>& qs, Eigen::VectorXi&& sigma) noexcept
		: RBMStateObj<ScalarType, RBMStateValue<ScalarType> >(qs), sigma_(std::move(sigma))
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValue(const RBM<ScalarType>& qs, const Eigen::VectorXi& sigma) noexcept
		: RBMStateObj<ScalarType, RBMStateValue<ScalarType> >(qs), sigma_(sigma)
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValue(const RBMStateValue<ScalarType>& rhs) = default;
	RBMStateValue(RBMStateValue<ScalarType>&& rhs) = default;

	RBMStateValue& operator=(const RBMStateValue<ScalarType>& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		sigma_ = rhs.sigma_;
		theta_ = rhs.theta_;
		return *this;
	}

	RBMStateValue& operator=(RBMStateValue<ScalarType>&& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		sigma_ = std::move(rhs.sigma_);
		theta_ = std::move(rhs.theta_);
		return *this;
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
	inline ScalarType thetaAt(int j) const
	{
		return theta_(j);
	}
	

	void flip(int k)
	{
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) -= 2.0*RealScalarType(sigma_(k))*(this->qs_.W(j,k));
		}
		sigma_(k) *= -1;
	}

	void flip(int k, int l)
	{
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) += -2.0*RealScalarType(sigma_(k))*(this->qs_.W(j,k))
				-2.0*RealScalarType(sigma_(l))*(this->qs_.W(j,l));
		}
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	template<std::size_t N>
	void flip(const std::array<int, N>& v)
	{
		for(int elt: v)
		{
			for(int j = 0; j < theta_.size(); j++)
			{
				theta_(j) -= 2.0*RealScalarType(sigma_(elt))*(this->qs_.W(j,elt));
			}
		}
		for(int elt: v)
		{
			sigma_(elt) *= -1;
		}
	}

	
	using RBMStateObj<ScalarType, RBMStateValue<ScalarType> >::logRatio;
	

	const Eigen::VectorXi& getSigma() const & { return sigma_; } 
	Eigen::VectorXi getSigma() && { return std::move(sigma_); } 

	const VectorType& getTheta() const & { return theta_; } 
	VectorType getTheta() && { return std::move(theta_); } 

	std::tuple<Eigen::VectorXi, VectorType> data() const
	{
		return std::make_tuple(sigma_, theta_);
	}
};


template<typename ScalarType>
class RBMStateRef
	: public RBMStateObj<ScalarType, RBMStateRef<ScalarType> >
{
public:
	using VectorType = typename RBM<ScalarType>::VectorType;
	using T = ScalarType;

private:
	const Eigen::VectorXi& sigma_;
	const VectorType& theta_;
public:
	
	RBMStateRef(const RBM<ScalarType>& qs, const Eigen::VectorXi& sigma, const VectorType& theta) noexcept
		: RBMStateObj<ScalarType, RBMStateRef<ScalarType> >(qs), sigma_(sigma), theta_(theta)
	{
	}

	inline int sigmaAt(int i) const
	{
		return sigma_(i);
	}
	inline ScalarType thetaAt(int j) const
	{
		return theta_(j);
	}

	Eigen::VectorXi getSigma() const
	{
		return sigma_;
	}

	const VectorType& getTheta() const&
	{
		return theta_;
	}

	VectorType getTheta() &&
	{
		return theta_;
	}

};

template<typename T>
struct is_reference_state_type<RBMStateRef<T> >: public std::true_type {};

} //namespace yannq
#endif//YANNQ_STATES_RBMSTATE_HPP
