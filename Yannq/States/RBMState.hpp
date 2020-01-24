#ifndef YANNQ_STATES_RBMSTATE_HPP
#define YANNQ_STATES_RBMSTATE_HPP

#include "Machines/RBM.hpp"
#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"

namespace yannq
{

template<typename Machine>
struct RBMStateValue;
template<typename Machine, bool is_const = true>
class RBMStateRef;

template<typename T, bool useBias>
class MachineStateTypes<RBM<T, useBias> >
{
public:
	using StateValue = RBMStateValue<RBM<T, useBias> >;
	using StateRef = RBMStateRef<RBM<T, useBias> >;
};

template<typename Machine, class Derived>
class RBMStateObj
{
protected:
	const Machine& qs_;
public:
	using ScalarType = typename Machine::ScalarType;

	RBMStateObj(const Machine& qs) noexcept
		: qs_(qs)
	{
	}

	ScalarType logRatio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		using std::exp;
		using std::cosh;
		using std::log;
		
		ScalarType res = -2.0*qs_.A(k)*ScalarType(sigmaAt(k));

		int m = qs_.getM();
		for(int j = 0; j < m; j ++)
		{
			res += logCosh(thetaAt(j)-2.0*ScalarType(sigmaAt(k))*qs_.W(j,k))
				-logCosh(thetaAt(j));
		}
		return res;
	}

	inline ScalarType ratio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		return std::exp(logRatio(k));
	}

	ScalarType logRatio(int k, int l) const //calc psi(sigma ^ k ^ l)/psi(sigma)
	{
		using std::exp;
		using std::cosh;
		ScalarType res = -2.0*qs_.A(k)*ScalarType(sigmaAt(k))-2.0*qs_.A(l)*ScalarType(sigmaAt(l));
		const int m = qs_.getM();

		for(int j = 0; j < m; j ++)
		{
			ScalarType t = thetaAt(j)-2.0*ScalarType(sigmaAt(k))*qs_.W(j,k)-2.0*ScalarType(sigmaAt(l))*qs_.W(j,l);
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
	}

	ScalarType ratio(int k, int l) const
	{
		return std::exp(logRatio(k,l));
	}

	template<std::size_t N>
	ScalarType logRatio(const std::array<int, N>& v) const
	{
		ScalarType res{};
		const int m = qs_.getM();
		for(int elt: v)
		{
			res -= 2.0*qs_.A(elt)*ScalarType(sigmaAt(elt));
		}
		for(int j = 0; j < m; j++)
		{
			ScalarType t = thetaAt(j);
			for(int elt: v)
			{
				t -= 2.0*ScalarType(sigmaAt(elt))*qs_.W(j,elt);
			}
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
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
	ScalarType logRatio(Eigen::VectorXi to) const
	{
		typename Machine::VectorType thetaTo = qs_.calcTheta(to);
		to -= static_cast<const Derived*>(this)->getSigma();
		ScalarType s = (qs_->getA().transpose())*(to.cast<ScalarType>());
		for(int j = 0; j < qs_->getM(); j++)
		{
			s += logCosh(thetaTo(j)) - logCosh(thetaAt(j));
		}
		return s;
	}

};

template<typename Machine>
struct RBMStateValue
	: public RBMStateObj<Machine, RBMStateValue<Machine> >
{
private:
	Eigen::VectorXi sigma_;
	typename Machine::VectorType theta_;

public:
	using VectorType=typename Machine::VectorType;
	using ScalarType = typename Machine::ScalarType;

	RBMStateValue(const Machine& qs, Eigen::VectorXi&& sigma) noexcept
		: RBMStateObj<Machine, RBMStateValue<Machine> >(qs), sigma_(std::move(sigma))
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValue(const Machine& qs, const Eigen::VectorXi& sigma) noexcept
		: RBMStateObj<Machine, RBMStateValue<Machine> >(qs), sigma_(sigma)
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValue(const RBMStateValue<Machine>& rhs) noexcept = default;
	RBMStateValue(RBMStateValue<Machine>&& rhs) noexcept = default;

	RBMStateValue& operator=(const RBMStateValue<Machine>& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		sigma_ = rhs.sigma_;
		theta_ = rhs.theta_;
		return *this;
	}

	RBMStateValue& operator=(RBMStateValue<Machine>&& rhs) noexcept
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
	
	template<std::size_t N>
	void flip(const std::array<int, N>& v)
	{
		for(int elt: v)
		{
			for(int j = 0; j < theta_.size(); j++)
			{
				theta_(j) -= 2.0*ScalarType(sigma_(elt))*(this->qs_.W(j,elt));
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
			theta_(j) += -2.0*ScalarType(sigma_(k))*(this->qs_.W(j,k))
				-2.0*ScalarType(sigma_(l))*(this->qs_.W(j,l));
		}
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	void flip(int k)
	{
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) -= 2.0*ScalarType(sigma_(k))*(this->qs_.W(j,k));
		}
		sigma_(k) *= -1;
	}
	
	using RBMStateObj<Machine, RBMStateValue<Machine> >::logRatio;
	ScalarType logRatio(const RBMStateValue& other)
	{
		ScalarType res = (this->qs_.getA().transpose())*
			(other.getSigma() - sigma_).template cast<ScalarType>();
		for(int j = 0; j < theta_.size(); j++)
		{
			res += logCosh(other.theta_(j)) - logCosh(theta_(j));
		}
		return res;
	}

	const Eigen::VectorXi& getSigma() const & { return sigma_; } 
	Eigen::VectorXi getSigma() && { return std::move(sigma_); } 

	const VectorType& getTheta() const & { return theta_; } 
	VectorType getTheta() && { return std::move(theta_); } 

	std::tuple<Eigen::VectorXi, VectorType> data() const
	{
		return std::make_tuple(sigma_, theta_);
	}
};


template<typename Machine, bool is_const>
class RBMStateRef
	: public RBMStateObj<Machine, RBMStateRef<Machine> >
{
public:
	using VectorType = typename Machine::VectorType;
private:
	typedef typename std::conditional<is_const, const Eigen::VectorXi, Eigen::VectorXi>::type SigmaType;
	typedef typename std::conditional<is_const, const VectorType, VectorType>::type ThetaType;
	SigmaType& sigma_;
	ThetaType& theta_;
public:
	
	using T = typename Machine::ScalarType;

	RBMStateRef(const Machine& qs, SigmaType& sigma, ThetaType& theta) noexcept
		: RBMStateObj<Machine, RBMStateRef<Machine, is_const> >(qs), sigma_(sigma), theta_(theta)
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

	typename Machine::VectorType getTheta() const
	{
		return theta_;
	}

};

template<typename T>
struct is_reference_state_type<RBMStateRef<T> >: public std::true_type {};

} //namespace yannq
#endif//YANNQ_STATES_RBMSTATE_HPP
