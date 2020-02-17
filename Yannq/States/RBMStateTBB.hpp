#ifndef YANNQ_STATES_RBMSTATETBB_HPP
#define YANNQ_STATES_RBMSTATETBB_HPP

#include "Machines/RBM.hpp"
#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "States/RBMState.hpp"
namespace yannq
{

template<typename Machine>
struct RBMStateValueTBB;

template<typename T, bool useBias>
class MachineStateTypesTBB<RBM<T, useBias> >
{
public:
	using StateValue = RBMStateValueTBB<RBM<T, useBias> >;
	using StateRef = RBMStateRef<RBM<T, useBias> >;
};

template<typename ValueType, class Func>
struct Sum
{
	ValueType value;
	Func f;
	Sum(): value{}
	{
	}
	Sum(Sum& s, tbb::split )
	{
		value = ValueType{};
	}
	void operator()(const blocked_range<int>& r)
	{
		float temp = value;
		for(int idx = r.begin(); idx != r.end(); ++idx)
		{
			temp += f(idx);
		}
		value = temp;
	}
	void join(Sum& rhs)
	{
		value += rhs.value;
	}
};

template<typename Machine, class Derived>
class RBMStateObjTBB
{
protected:
	const Machine& qs_;
public:
	using T = typename Machine::ScalarType;

	RBMStateObjTBB(const Machine& qs) noexcept
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

		auto f = [&](int idx)
		{
			return logCosh(thetaAt(idx)-2.0*T(sigmaAt(k))*qs_.W(idx,k))
					-logCosh(thetaAt(idx));
		};

		Sum<T, decltype(f)> sum;
		tbb::parallel_reduce(tbb::blocked_range<int>(0, m), f);

		res += sum.value;
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

		auto f = [&](int idx)
		{
			T t = thetaAt(idx)-2.0*T(sigmaAt(k))*qs_.W(idx,k)-2.0*T(sigmaAt(l))*qs_.W(idx,l);
			return logCosh(t)-logCosh(thetaAt(idx));
		};

		Sum<T, decltype(f)> sum;
		tbb::parallel_reduce(tbb::blocked_range<int>(0, m), f);

		res += sum.value;
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

		auto f = [&](int idx)
		{
			T t = thetaAt(idx);
			for(int elt: v)
			{
				t -= 2.0*T(sigmaAt(elt))*qs_.W(idx,elt);
			}
			return logCosh(t)-logCosh(thetaAt(idx));
		};

		Sum<T, decltype(f)> sum;
		tbb::parallel_reduce(tbb::blocked_range<int>(0, m), f);

		res += sum.value;
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

	const Machine& getRBM() const
	{
		return qs_;
	}
	
	template<class Derived>
	T logRatio(const RBMStateObjTBB<Machine, Derived<Machine> >& other)
	{
		T res = (qs_.getA().transpose())*
			(other.getSigma() - getSigma()).template cast<T>();

		auto f = [&](int idx)
		{
			return logCosh(other.thetaAt(idx)) - logCosh(thetaAt(idx));
		}

		Sum<T, decltype(f)> sum;
		tbb::parallel_reduce(tbb::blocked_range<int>(0, m), f);
		res += sum.value;
		return res;
	}

	inline Eigen::VectorXi getSigma()
	{
		return static_cast<const Derived*>(this)->getSigma();
	}

};

template<typename Machine>
struct RBMStateValueTBB
	: public RBMStateObjTBB<Machine, RBMStateValueTBB<Machine> >
{
private:
	Eigen::VectorXi sigma_;
	typename Machine::VectorType theta_;

public:
	using Vector=typename Machine::VectorType;
	using T = typename Machine::ScalarType;

	RBMStateValueTBB(const Machine& qs, Eigen::VectorXi&& sigma) noexcept
		: RBMStateObjTBB<Machine, RBMStateValueTBB<Machine> >(qs), sigma_(std::move(sigma))
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValueTBB(const Machine& qs, const Eigen::VectorXi& sigma) noexcept
		: RBMStateObjTBB<Machine, RBMStateValueTBB<Machine> >(qs), sigma_(sigma)
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValueTBB(const RBMStateValueTBB<Machine>& rhs) = default;
	RBMStateValueTBB(RBMStateValueTBB<Machine>&& rhs) = default;

	RBMStateValueTBB& operator=(const RBMStateValueTBB<Machine>& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		sigma_ = rhs.sigma_;
		theta_ = rhs.theta_;
		return *this;
	}

	RBMStateValueTBB& operator=(RBMStateValueTBB<Machine>&& rhs) noexcept
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
			tbb::parallel_for(0u, theta_.size(), [&](std::size_t idx)
			{
				theta_(idx) -= 2.0*T(sigma_(elt))*(this->qs_.W(idx,elt));
			});
		}
		for(int elt: v)
		{
			sigma_(elt) *= -1;
		}
	}

	void flip(int k, int l)
	{
		tbb::parallel_for(0u, theta_.size(), [&](std::size_t idx)
		{
			theta_(idx) += -2.0*T(sigma_(k))*(this->qs_.W(idx,k))
				-2.0*T(sigma_(l))*(this->qs_.W(idx,l));
		});
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	void flip(int k)
	{
		tbb::parallel_for(0u, theta_.size(), [&](std::size_t idx)
		{
			theta_(j) -= 2.0*T(sigma_(k))*(this->qs_.W(j,k));
		});
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
#endif//YANNQ_STATES_RBMSTATETBB_HPP
