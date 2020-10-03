#ifndef YANNQ_STATES_RBMSTATEMT_HPP
#define YANNQ_STATES_RBMSTATEMT_HPP

#include "Machines/RBM.hpp"
#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "States/RBMState.hpp"

#include "./utils.hpp"

namespace yannq
{

template<typename Machine>
struct RBMStateValueMT;

template<typename T>
class MachineStateTypesMT<RBM<T> >
{
public:
	using StateValue = RBMStateValueMT<RBM<T> >;
	using StateRef = RBMStateRef<RBM<T> >;
};

template<typename T, class Derived>
class RBMStateObjMT
{
protected:
	const RBM<T>& qs_;
public:
	using Scalar = T;
	using RealScalar = remove_complex_t<Scalar>;
	using Machine = RBM<Scalar>;

	RBMStateObjMT(const Machine& qs) noexcept
		: qs_(qs)
	{
	}
	
	/********************** logRatio for a single spin ************************/

	/// logRatio for complex Scalar
	/// returns log[psi(sigma ^ k)] - log[psi(sigma)]
	Scalar logRatio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		using std::exp;
		using std::cosh;
		using std::log;
		using Range = tbb::blocked_range<uint32_t>;
		
		Scalar res = -2.0*qs_.A(k)*RealScalar(sigmaAt(k));

		const uint32_t m = qs_.getM();

		res += tbb::parallel_reduce(Range(0u, m), Scalar{}, 
			[this, k](const Range& r, Scalar val) -> Scalar
		{
			for(uint32_t idx = r.begin(); idx != r.end(); ++idx)
			{
				Scalar t = thetaAt(idx)
					-2.0*Scalar(sigmaAt(k))*qs_.W(idx,k);
				val += logCosh(t) - logCosh(thetaAt(idx));
			}
			return val;
		}, std::plus<Scalar>());
		return res;
	}

	/// logRatioRe for complex Scalar
	/// returns real part of log[psi(sigma ^ k)] - log[psi(sigma)]
	template<typename U = Scalar,
		typename std::enable_if<is_complex_type<U>::value, int>::type = 0>
	RealScalar logRatioRe(int k) const 
	{
		return std::real(logRatio(k));
	}

	/// logRatioRe for real Scalar
	template<typename U = Scalar, 
		typename std::enable_if<!is_complex_type<U>::value, int>::type = 0>
	inline Scalar logRatioRe(int k) const
	{
		return logRatio(k);
	}
	
	/************************ logRatio for two spins **************************/

	/// logRatio for Scalar
	/// returns log[psi(sigma ^ k ^ l)] - log[psi(sigma)]
	Scalar logRatio(int k, int l) const 
	{
		using std::exp;
		using std::cosh;
		using Range = tbb::blocked_range<uint32_t>;

		Scalar res = -2.0*qs_.A(k)*RealScalar(sigmaAt(k))
			-2.0*qs_.A(l)*RealScalar(sigmaAt(l));

		const uint32_t m = qs_.getM();

		res += tbb::parallel_reduce(Range(0u, m), Scalar{}, 
			[this, k, l](const Range& r, Scalar val) -> Scalar
		{
			for(uint32_t idx = r.begin(); idx != r.end(); ++idx)
			{
				Scalar t = thetaAt(idx)
					-2.0*Scalar(sigmaAt(k))*qs_.W(idx,k)
					-2.0*Scalar(sigmaAt(l))*qs_.W(idx,l);
				val += logCosh(t)-logCosh(thetaAt(idx));
			}
			return val;
		}, std::plus<Scalar>());
		return res;
	}

	/// logRatioRe for complex Scalar
	template<typename U = Scalar, 
		typename std::enable_if<is_complex_type<U>::value, int>::type = 0>
	RealScalar logRatioRe(int k, int l) const
	{
		return std::real(logRatio(k,l));
	}

	/// logRatioRe for real Scalar
	template<typename U = Scalar,
		typename std::enable_if<!is_complex_type<U>::value, int>::type = 0>
	inline RealScalar logRatioRe(int k, int l) const
	{
		return logRatio(k, l);
	}

	/************************ logRatio for vector ******************************/

	/// logRatio for Scalar
	Scalar logRatio(const std::vector<int>& v) const
	{
		using Range = tbb::blocked_range<uint32_t>;

		Scalar res{};
		const uint32_t m = qs_.getM();
		for(int elt: v)
		{
			res -= 2.0*qs_.A(elt)*RealScalar(sigmaAt(elt));
		}

		res += tbb::parallel_reduce(Range(0u, m), Scalar{},
			[this, &v](const Range& r, Scalar val) -> Scalar
		{
			for(uint32_t idx = r.begin(); idx != r.end(); ++idx)
			{
				Scalar t = thetaAt(idx);
				for(int elt: v)
				{
					t -= 2.0*RealScalar(sigmaAt(elt))*qs_.W(idx,elt);
				}
				val += logCosh(t)-logCosh(thetaAt(idx));
			}
			return val;
		}, std::plus<Scalar>());
		return res;
	}

	/// logRatioRe for complex Scalar
	template<typename U = Scalar, 
		typename std::enable_if<is_complex_type<U>::value, int>::type = 0>
	RealScalar logRatioRe(const std::vector<int>& v) const
	{
		return std::real(logRatio(v));
	}
	
	//logRatioRe for real Scalar
	template<typename U = Scalar,
		typename std::enable_if<!is_complex_type<U>::value, int>::type = 0>
	inline Scalar logRatioRe(const std::vector<int>& v) const
	{
		return logRatio(v);
	}

	/********************* logRatio with another StateObj *********************/

	Scalar logRatio(const RBMStateObjMT<Scalar, Derived>& other)
	{
		using Range = tbb::blocked_range<uint32_t>;
		Scalar res = (qs_.getA().transpose())*
			(other.getSigma() - getSigma()).template cast<Scalar>();

		const uint32_t m = qs_.getM();
		res += tbb::parallel_reduce(Range(0u, m), Scalar{},
			[this, &other](const Range& r, Scalar val) -> Scalar
		{
			for(uint32_t idx = r.begin(); idx != r.end(); ++idx)
			{
				val += logCosh(other.thetaAt(idx)) - logCosh(thetaAt(idx));
			}
			return val;
		}, std::plus<Scalar>());
		return res;
	}

	/// logRatioRe for complex Scalar
	template<typename U = Scalar, 
		typename std::enable_if<is_complex_type<U>::value, int>::type = 0>
	RealScalar logRatioRe(const RBMStateObjMT<Scalar, Derived>& other)
	{
		return std::real(logRatio(other));
	}

	/// logRatioRe for real Scalar
	template<typename U = Scalar,
		typename std::enable_if<!is_complex_type<U>::value, int>::type = 0>
	Scalar logRatioRe(const RBMStateObjMT<Scalar, Derived>& other)
	{
		return logRatio(other);
	}

	/*********************** other utiliy functions ***************************/

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

template<typename Scalar>
struct RBMStateValueMT
	: public RBMStateObjMT<Scalar, RBMStateValueMT<Scalar> >
{
private:
	Eigen::VectorXi sigma_;
	typename RBM<Scalar>::Vector theta_;

public:
	using Machine = RBM<Scalar>;
	using Vector = typename Machine::Vector;
	using RealScalar = remove_complex_t<Scalar>;

	RBMStateValueMT(const Machine& qs, Eigen::VectorXi&& sigma) noexcept
		: RBMStateObjMT<Scalar, RBMStateValueMT<Scalar> >(qs), 
		sigma_(std::move(sigma))
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValueMT(const Machine& qs, const Eigen::VectorXi& sigma) noexcept
		: RBMStateObjMT<Scalar, RBMStateValueMT<Scalar> >(qs), sigma_(sigma)
	{
		theta_ = this->qs_.calcTheta(sigma_);
	}

	RBMStateValueMT(const RBMStateValueMT<Scalar>& rhs) noexcept = default;
	RBMStateValueMT(RBMStateValueMT<Scalar>&& rhs) noexcept = default;

	RBMStateValueMT& operator=(const RBMStateValueMT<Scalar>& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		sigma_ = rhs.sigma_;
		theta_ = rhs.theta_;
		return *this;
	}

	RBMStateValueMT& operator=(RBMStateValueMT<Scalar>&& rhs) noexcept
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
	inline Scalar thetaAt(int j) const
	{
		return theta_(j);
	}

	void flip(int k)
	{
		tbb::parallel_for(0u, uint32_t(theta_.size()), 
			[this, k](uint32_t idx)
		{
			theta_(idx) -= 2.0*RealScalar(sigma_(k))*(this->qs_.W(idx,k));
		});
		sigma_(k) *= -1;
	}

	void flip(int k, int l)
	{
		tbb::parallel_for(0u, uint32_t(theta_.size()), 
			[this, k, l](uint32_t idx)
		{
			theta_(idx) += -2.0*RealScalar(sigma_(k))*(this->qs_.W(idx,k))
				-2.0*RealScalar(sigma_(l))*(this->qs_.W(idx,l));
		});
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	template<std::size_t N>
	void flip(const std::array<int, N>& v)
	{
		tbb::parallel_for(0u, uint32_t(theta_.size()), 
			[this, &v](uint32_t idx)
		{
			Scalar diff{};
			for(int elt: v)
			{
				 diff += 2.0*RealScalar(sigma_(elt))*(this->qs_.W(idx,elt));
			}
			theta_(idx) -= diff;
		});
		for(int elt: v)
		{
			sigma_(elt) *= -1;
		}
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
