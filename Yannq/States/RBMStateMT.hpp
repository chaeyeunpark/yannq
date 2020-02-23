#ifndef YANNQ_STATES_RBMSTATEMT_HPP
#define YANNQ_STATES_RBMSTATEMT_HPP

#include "Machines/RBM.hpp"
#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"

#include "States/RBMState.hpp"
namespace yannq
{

template<typename Machine>
struct RBMStateValueMT;

template<typename T, bool useBias>
class MachineStateTypesMT<RBM<T, useBias> >
{
public:
	using StateValue = RBMStateValueMT<RBM<T, useBias> >;
	using StateRef = RBMStateRef<RBM<T, useBias> >;
};

template<typename Machine, class Derived>
class RBMStateObjMT
{
protected:
	const Machine& qs_;
public:
	using T = typename Machine::ScalarType;
	using RealT = typename remove_complex<T>::type;

	RBMStateObjMT(const Machine& qs) noexcept
		: qs_(qs)
	{
	}

	T logRatio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		using std::exp;
		using std::cosh;
		using std::log;
		
		T res = -2.0*qs_.A(k)*T(sigmaAt(k));

		int m = qs_.getM();

		RealT re{};
		RealT im{};

#pragma omp parallel for schedule(static, 4) reduction(+:re,im)
		for(int j = 0; j < m; j ++)
		{
			T r = logCosh(thetaAt(j)-2.0*T(sigmaAt(k))*qs_.W(j,k))
				-logCosh(thetaAt(j));
			re += std::real(r);
			im += std::imag(r);
		}
		res += T{re, im};
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
		T res = -2.0*qs_.A(k)*T(sigmaAt(k))-2.0*qs_.A(l)*T(sigmaAt(l));
		const int m = qs_.getM();

		RealT re{};
		RealT im{};
#pragma omp parallel for schedule(static, 4) reduction(+:re,im)
		for(int j = 0; j < m; j ++)
		{
			T t = thetaAt(j)-2.0*T(sigmaAt(k))*qs_.W(j,k)-2.0*T(sigmaAt(l))*qs_.W(j,l);
			T r = logCosh(t)-logCosh(thetaAt(j));
			re += std::real(r);
			im += std::imag(r);
		}
		res += T{re,im};
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
		const int m = qs_.getM();
		for(int elt: v)
		{
			res -= 2.0*qs_.A(elt)*T(sigmaAt(elt));
		}

		RealT re{};
		RealT im{};

#pragma omp parallel for schedule(static,4) reduction(+:re, im)
		for(int j = 0; j < m; j++)
		{
			T t = thetaAt(j);
			for(int elt: v)
			{
				t -= 2.0*T(sigmaAt(elt))*qs_.W(j,elt);
			}
			T r = logCosh(t)-logCosh(thetaAt(j));
			re += std::real(r);
			im += std::imag(r);
		}
		res += T{re,im};
		return res;
	}

	T logRatio(const RBMStateObjMT<Machine, Derived >& other)
	{
		T res = (qs_.getA().transpose())*
			(other.getSigma() - getSigma()).template cast<T>();

		const int m = qs_.getM();
		RealT re{};
		RealT im{};

#pragma omp parallel for schedule(static, 4) reduction(+:re, im)
		for(int j = 0; j < m; j++)
		{
			T r = logCosh(other.thetaAt(j)) - logCosh(thetaAt(j));
			re += std::real(r);
			im += std::imag(r);
		};

		res += T{re,im};
		return res;
	}

	inline Eigen::VectorXi getSigma()
	{
		return static_cast<const Derived*>(this)->getSigma();
	}

	inline int sigmaAt(int i) const
	{
		return static_cast<const Derived*>(this)->sigmaAt(i);
	}

	inline T thetaAt(int i) const
	{
		return static_cast<const Derived*>(this)->thetaAt(i);
	}

	const Machine& getRBM() const
	{
		return qs_;
	}
};

template<typename Machine>
struct RBMStateValueMT
	: public RBMStateObjMT<Machine, RBMStateValueMT<Machine> >
{
private:
	Eigen::VectorXi sigma_;
	typename Machine::VectorType theta_;

public:
	using Vector=typename Machine::VectorType;
	using T = typename Machine::ScalarType;

	RBMStateValueMT(const Machine& qs, Eigen::VectorXi&& sigma) noexcept
		: RBMStateObjMT<Machine, RBMStateValueMT<Machine> >(qs), sigma_(std::move(sigma))
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValueMT(const Machine& qs, const Eigen::VectorXi& sigma) noexcept
		: RBMStateObjMT<Machine, RBMStateValueMT<Machine> >(qs), sigma_(sigma)
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValueMT(const RBMStateValueMT<Machine>& rhs) = default;
	RBMStateValueMT(RBMStateValueMT<Machine>&& rhs) = default;

	RBMStateValueMT& operator=(const RBMStateValueMT<Machine>& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		sigma_ = rhs.sigma_;
		theta_ = rhs.theta_;
		return *this;
	}

	RBMStateValueMT& operator=(RBMStateValueMT<Machine>&& rhs) noexcept
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
	inline T thetaAt(int j) const
	{
		return theta_(j);
	}
	
	template<std::size_t N>
	void flip(const std::array<int, N>& v)
	{
		for(int elt: v)
		{
#pragma omp parallel for schedule(static,4)
			for(int j = 0; j < theta_.size(); j++)
			{
				theta_(j) -= 2.0*T(sigma_(elt))*(this->qs_.W(j,elt));
			}
		}
		for(int elt: v)
		{
			sigma_(elt) *= -1;
		}
	}

	void flip(int k, int l)
	{
#pragma omp parallel for schedule(static,4)
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) += -2.0*T(sigma_(k))*(this->qs_.W(j,k))
				-2.0*T(sigma_(l))*(this->qs_.W(j,l));
		}
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	void flip(int k)
	{
#pragma omp parallel for schedule(static,4)
		for(int j = 0; j < theta_.size(); j++)
		{
			theta_(j) -= 2.0*T(sigma_(k))*(this->qs_.W(j,k));
		}
		sigma_(k) *= -1;
	}
	
	const Eigen::VectorXi& getSigma() const & { return sigma_; } 
	Eigen::VectorXi getSigma() && { return std::move(sigma_); } 

	const Vector& getTheta() const & { return theta_; } 
	Vector getTheta() && { return std::move(theta_); } 

	std::tuple<Eigen::VectorXi, Vector> data() const
	{
		return std::make_tuple(sigma_, theta_);
	}
};

} //namespace yannq
#endif//YANNQ_STATES_RBMSTATEMT_HPP
