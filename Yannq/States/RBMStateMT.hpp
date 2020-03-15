#ifndef YANNQ_STATES_RBMSTATEMT_HPP
#define YANNQ_STATES_RBMSTATEMT_HPP

#include <vector>

#include "Machines/RBM.hpp"
#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"

#include "States/RBMState.hpp"
namespace yannq
{

template<typename T>
class RBMStateValueMT;

template<typename T>
class MachineStateTypesMT<RBM<T> >
{
public:
	using StateValue = RBMStateValueMT<RBM<T> >;
	using StateRef = RBMStateRef<RBM<T> >;
};

template<typename ScalarType, class Derived>
class RBMStateObjMT
{
protected:
	const RBM<ScalarType>& qs_;

public:
	using Machine = RBM<ScalarType>;
	using RealScalarType = typename remove_complex<ScalarType>::type;

	RBMStateObjMT(const Machine& qs) noexcept
		: qs_(qs)
	{
	}
	
	/********************** logRatio for a single spin ************************/

	/// logRatio for complex ScalarType
	/// returns log[psi(sigma ^ k)] - log[psi(sigma)]
	template<typename T = ScalarType, std::enable_if_t<is_complex_type<T>::value, int> = 0>
	ScalarType logRatio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		using std::exp;
		using std::cosh;
		using std::log;
		
		ScalarType res = -2.0*qs_.A(k)*ScalarType(sigmaAt(k));

		int m = qs_.getM();

		RealScalarType re{};
		RealScalarType im{};

#pragma omp parallel for schedule(static, 4) reduction(+:re,im)
		for(int j = 0; j < m; j ++)
		{
			ScalarType r = logCosh(thetaAt(j)-2.0*ScalarType(sigmaAt(k))*qs_.W(j,k))
				-logCosh(thetaAt(j));
			re += std::real(r);
			im += std::imag(r);
		}
		res += ScalarType{re, im};
		return res;
	}

	/// logRatioRe for complex ScalarType
	/// returns real part of log[psi(sigma ^ k)] - log[psi(sigma)]
	template<typename T = ScalarType, std::enable_if_t<is_complex_type<T>::value, int> = 0>
	RealScalarType logRatioRe(int k) const 
	{
		using std::exp;
		using std::cosh;
		using std::log;
		
		RealScalarType res = -2.0*qs_.A(k).real()*RealScalarType(sigmaAt(k));

		int m = qs_.getM();

#pragma omp parallel for schedule(static, 4) reduction(+:res)
		for(int j = 0; j < m; j ++)
		{
			ScalarType r = logCosh(thetaAt(j)-2.0*ScalarType(sigmaAt(k))*qs_.W(j,k))
				-logCosh(thetaAt(j));
			res += std::real(r);
		}
		return res;
	}

	/// logRatio for real ScalarType
	template<typename T = ScalarType, std::enable_if_t<!is_complex_type<T>::value, int> = 0>
	inline ScalarType logRatioRe(int k) const
	{
		return logRatio(k);
	}

	/// logRatio for real ScalarType
	template<typename T = ScalarType, std::enable_if_t<!is_complex_type<T>::value, int> = 0>
	ScalarType logRatio(int k) const
	{
		using std::exp;
		using std::cosh;
		using std::log;
		
		ScalarType res = -2.0*qs_.A(k)*T(sigmaAt(k));

		int m = qs_.getM();

#pragma omp parallel for schedule(static, 4) reduction(+:res)
		for(int j = 0; j < m; j ++)
		{
			res += logCosh(thetaAt(j)-2.0*RealScalarType(sigmaAt(k))*qs_.W(j,k))
				-logCosh(thetaAt(j));
		}
		return res;
	}

	inline ScalarType ratio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		return std::exp(logRatio(k));
	}

	
	/************************ logRatio for two spins **************************/

	/// logRatio for complex ScalarType
	template<typename T = ScalarType, std::enable_if_t<is_complex_type<T>::value, int> = 0>
	ScalarType logRatio(int k, int l) const //calc psi(sigma ^ k ^ l)/psi(sigma)
	{
		using std::exp;
		using std::cosh;
		ScalarType res = -2.0*qs_.A(k)*RealScalarType(sigmaAt(k))
			-2.0*qs_.A(l)*RealScalarType(sigmaAt(l));
		const int m = qs_.getM();

		RealScalarType re{};
		RealScalarType im{};
#pragma omp parallel for schedule(static, 4) reduction(+:re,im)
		for(int j = 0; j < m; j ++)
		{
			T t = thetaAt(j)-2.0*RealScalarType(sigmaAt(k))*qs_.W(j,k)
				-2.0*RealScalarType(sigmaAt(l))*qs_.W(j,l);
			T r = logCosh(t)-logCosh(thetaAt(j));
			re += std::real(r);
			im += std::imag(r);
		}
		res += T{re,im};
		return res;
	}

	/// logRatioRe for complex ScalarType
	template<typename T = ScalarType, std::enable_if_t<is_complex_type<T>::value, int> = 0>
	RealScalarType logRatioRe(int k, int l) const
	{
		using std::exp;
		using std::cosh;
		RealScalarType res = -2.0*qs_.A(k).real()*RealScalarType(sigmaAt(k))
			-2.0*qs_.A(l).real()*RealScalarType(sigmaAt(l));
		const int m = qs_.getM();

#pragma omp parallel for schedule(static, 4) reduction(+:res)
		for(int j = 0; j < m; j ++)
		{
			T t = thetaAt(j)-2.0*RealScalarType(sigmaAt(k))*qs_.W(j,k)
				-2.0*RealScalarType(sigmaAt(l))*qs_.W(j,l);
			T r = logCosh(t)-logCosh(thetaAt(j));
			res += std::real(r);
		}
		return res;
	}
	
	/// logRatio for real ScalarType
	template<typename T = ScalarType, std::enable_if_t<!is_complex_type<T>::value, int> = 0>
	ScalarType logRatio(int k, int l) const 
	{
		using std::exp;
		using std::cosh;
		ScalarType res = -2.0*qs_.A(k)*RealScalarType(sigmaAt(k))
			-2.0*qs_.A(l)*RealScalarType(sigmaAt(l));
		const int m = qs_.getM();

#pragma omp parallel for schedule(static, 4) reduction(+:res)
		for(int j = 0; j < m; j ++)
		{
			T t = thetaAt(j)-2.0*T(sigmaAt(k))*qs_.W(j,k)-2.0*T(sigmaAt(l))*qs_.W(j,l);
			res += logCosh(t)-logCosh(thetaAt(j));
		}
		return res;
	}

	/// logRatioRe for real ScalarType
	template<typename T = ScalarType, std::enable_if_t<!is_complex_type<T>::value, int> = 0>
	inline ScalarType logRatioRe(int k, int l) const
	{
		return logRatio(k, l);
	}

	ScalarType ratio(int k, int l) const
	{
		return std::exp(logRatio(k,l));
	}


	/************************ logRatio for vector ******************************/

	//logRatio for complex ScalarType
	template<typename T = ScalarType, std::enable_if_t<is_complex_type<T>::value, int> = 0>
	ScalarType logRatio(const std::vector<int>& v) const
	{
		ScalarType res{};
		const int m = qs_.getM();
		for(int elt: v)
		{
			res -= 2.0*qs_.A(elt)*RealScalarType(sigmaAt(elt));
		}

		RealScalarType re{};
		RealScalarType im{};

#pragma omp parallel for schedule(static,4) reduction(+:re, im)
		for(int j = 0; j < m; j++)
		{
			T t = thetaAt(j);
			for(int elt: v)
			{
				t -= 2.0*RealScalarType(sigmaAt(elt))*qs_.W(j,elt);
			}
			T r = logCosh(t)-logCosh(thetaAt(j));
			re += std::real(r);
			im += std::imag(r);
		}
		res += T{re,im};
		return res;
	}

	/// logRatio for Real ScalarType
	template<typename T = ScalarType, std::enable_if_t<!is_complex_type<T>::value, int> = 0>
	ScalarType logRatio(const std::vector<int>& v) const
	{
		ScalarType res{};
		const int m = qs_.getM();
		for(int elt: v)
		{
			res -= 2.0*qs_.A(elt)*RealScalarType(sigmaAt(elt));
		}


#pragma omp parallel for schedule(static,4) reduction(+:res)
		for(int j = 0; j < m; j++)
		{
			T t = thetaAt(j);
			for(int elt: v)
			{
				t -= 2.0*RealScalarType(sigmaAt(elt))*qs_.W(j,elt);
			}
			T r = logCosh(t)-logCosh(thetaAt(j));
			res += r;
		}
		return res;
	}

	/// logRatioRe for complex ScalarType
	template<typename T = ScalarType, std::enable_if_t<is_complex_type<T>::value, int> = 0>
	RealScalarType logRatioRe(const std::vector<int>& v) const
	{
		RealScalarType res{};
		const int m = qs_.getM();
		for(int elt: v)
		{
			res -= 2.0*qs_.A(elt).real()*RealScalarType(sigmaAt(elt));
		}

#pragma omp parallel for schedule(static,4) reduction(+:res)
		for(int j = 0; j < m; j++)
		{
			T t = thetaAt(j);
			for(int elt: v)
			{
				t -= 2.0*RealScalarType(sigmaAt(elt))*qs_.W(j,elt);
			}
			T r = logCosh(t)-logCosh(thetaAt(j));
			res += std::real(r);
		}
		return res;
	}
	
	//logRatioRe for real ScalarType
	template<typename T = ScalarType, std::enable_if_t<!is_complex_type<T>::value, int> = 0>
	inline ScalarType logRatioRe(const std::vector<int>& v) const
	{
		return logRatio(v);
	}


	/********************* logRatio with another StateObj *********************/

	/// logRatio for complex ScalarType
	template<std::size_t N, typename T = ScalarType, std::enable_if_t<is_complex_type<T>::value, int> = 0>
	ScalarType logRatio(const RBMStateObjMT<ScalarType, Derived>& other)
	{
		ScalarType res = (qs_.getA().transpose())*
			(other.getSigma() - getSigma()).template cast<RealScalarType>();

		const int m = qs_.getM();
		RealScalarType re{};
		RealScalarType im{};

#pragma omp parallel for schedule(static, 4) reduction(+:re, im)
		for(int j = 0; j < m; j++)
		{
			ScalarType r = logCosh(other.thetaAt(j)) - logCosh(thetaAt(j));
			re += std::real(r);
			im += std::imag(r);
		};

		res += ScalarType{re,im};
		return res;
	}

	/// logRatio for real ScalarType
	template<std::size_t N, typename T = ScalarType, std::enable_if_t<!is_complex_type<T>::value, int> = 0>
	ScalarType logRatio(const RBMStateObjMT<ScalarType, Derived>& other)
	{
		ScalarType res = (qs_.getA().transpose())*
			(other.getSigma() - getSigma()).template cast<RealScalarType>();

		const int m = qs_.getM();

#pragma omp parallel for schedule(static, 4) reduction(+:res)
		for(int j = 0; j < m; j++)
		{
			res += logCosh(other.thetaAt(j)) - logCosh(thetaAt(j));
		};
		return res;
	}

	/// logRatioRe for complex ScalarType
	template<std::size_t N, typename T = ScalarType, std::enable_if_t<is_complex_type<T>::value, int> = 0>
	ScalarType logRatioRe(const RBMStateObjMT<ScalarType, Derived>& other)
	{
		RealScalarType res = (qs_.getA().transpose()).real()*
			(other.getSigma() - getSigma()).template cast<RealScalarType>();

		const int m = qs_.getM();

#pragma omp parallel for schedule(static, 4) reduction(+:res)
		for(int j = 0; j < m; j++)
		{
			ScalarType r = logCosh(other.thetaAt(j)) - logCosh(thetaAt(j));
			res += std::real(r);
		};
		return res;
	}

	/// logRatioRe for real ScalarType
	template<std::size_t N, typename T = ScalarType, std::enable_if_t<!is_complex_type<T>::value, int> = 0>
	ScalarType logRatioRe(const RBMStateObjMT<ScalarType, Derived>& other)
	{
		return logRatio(other);
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
class RBMStateValueMT
	: public RBMStateObjMT<ScalarType, RBMStateValueMT<ScalarType> >
{
public:
	using VectorType = typename RBM<ScalarType>::VectorType;
	using RealScalarType = typename remove_complex<ScalarType>::type;
	using Machine = RBM<ScalarType>;

private:
	Eigen::VectorXi sigma_;
	VectorType theta_;

public:

	RBMStateValueMT(const Machine& qs, Eigen::VectorXi&& sigma) noexcept
		: RBMStateObjMT<ScalarType, RBMStateValueMT<ScalarType> >(qs), sigma_(std::move(sigma))
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValueMT(const Machine& qs, const Eigen::VectorXi& sigma) noexcept
		: RBMStateObjMT<ScalarType, RBMStateValueMT<ScalarType> >(qs), sigma_(sigma)
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValueMT(const RBMStateValueMT<ScalarType>& rhs) = default;
	RBMStateValueMT(RBMStateValueMT<ScalarType>&& rhs) = default;

	RBMStateValueMT& operator=(const RBMStateValueMT<ScalarType>& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		sigma_ = rhs.sigma_;
		theta_ = rhs.theta_;
		return *this;
	}

	RBMStateValueMT& operator=(RBMStateValueMT<ScalarType>&& rhs) noexcept
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
#pragma omp parallel for schedule(static,4)
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) -= 2.0*RealScalarType(sigma_(k))*(this->qs_.W(j,k));
		}
		sigma_(k) *= -1;
	}

	void flip(int k, int l)
	{
#pragma omp parallel for schedule(static,4)
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
#pragma omp parallel for schedule(static,4)
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

	
	const Eigen::VectorXi& getSigma() const & { return sigma_; } 
	Eigen::VectorXi getSigma() && { return std::move(sigma_); } 

	const VectorType& getTheta() const & { return theta_; } 
	VectorType getTheta() && { return std::move(theta_); } 

	std::tuple<Eigen::VectorXi, VectorType> data() const
	{
		return std::make_tuple(sigma_, theta_);
	}
};

} //namespace yannq
#endif//YANNQ_STATES_RBMSTATEMT_HPP
