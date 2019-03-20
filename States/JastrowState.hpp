#ifndef CY_NNQS_JASTROW_SAMPLES_HPP
#define CY_NNQS_JASTROW_SAMPLES_HPP
#include "Machines/Jastrow.hpp"
#include "Utilities/type_traits.hpp"

namespace nnqs
{

template<typename T>
class JastrowState;

template<typename T>
class MachineStateTypes<Jastrow<T> >
{
public:
	using StateValue = JastrowState<T>;
	using StateRef = JastrowState<T>;
};

template<typename ScalarT>
class JastrowState
{
private:
	Jastrow<ScalarT>& qs_;
	Eigen::VectorXi sigma_;
	ScalarT theta_;

public:
	using T = ScalarT;

	JastrowState(Jastrow<T>& qs, const Eigen::VectorXi& sigma)
		: qs_(qs), sigma_(sigma)
	{
		theta_ = qs.calcTheta(sigma_);
	}
	JastrowState(Jastrow<T>& qs, const Eigen::VectorXi& sigma, T theta)
		: qs_(qs), sigma_(sigma), theta_(theta)
	{
	}

	JastrowState(const JastrowState<T>& rhs)
		: qs_(rhs.qs_), sigma_(rhs.sigma_), theta_(rhs.theta_)
	{
	}

	JastrowState<T> operator=(const JastrowState<T>& rhs)
	{
		assert(&rhs.qs_ == &qs_);

		sigma_ = rhs.sigma_;
		theta_ = rhs.theta_;
	}

	T logRatio(int k) const
	{
		T res = -2.0*qs_.A(k)*T(sigma_(k));
		for(int j = k+1; j < qs_.getN(); j++)
		{
			res -= 2.0*qs_.J(k,j)*T(sigma_(k)*sigma_(j));
		}
		for(int i = 0; i < k; i++)
		{
			res -= 2.0*qs_.J(i,k)*T(sigma_(k)*sigma_(i));
		}
		return res;
	}

	T logRatio(int k, int l) const
	{
		T res = logRatio(k);
		res += logRatio(l);
		res += 4.0*(qs_.J(k,l)+qs_.J(l,k))*T(sigma_(k)*sigma_(l));
		return res;
	}

	T ratio(int k) const
	{
		return std::exp(logRatio(k));
	}

	T ratio(int k, int l) const
	{
		return std::exp(logRatio(k, l));
	}

	void flip(int k)
	{
		theta_ += logRatio(k);
		sigma_(k) *= -1;
	}

	void flip(int k, int l)
	{
		theta_ += logRatio(k,l);
		sigma_(k) *= -1;
		sigma_(l) *= -1;
	}

	template<std::size_t N>
	void flip(const std::array<int, N>& v)
	{
		for(auto a : v)
		{
			sigma_(a) *= -1;
		}
		theta_ = qs_.calcTheta(sigma_);
	}

	template<std::size_t N>
	T logRatio(const std::array<int, N>& v) const
	{
		Eigen::VectorXi toSigma = sigma_;
		for(auto a : v)
		{
			toSigma(a) *= -1;
		}
		return logRatio(toSigma);
	}

	T logRatio(const Eigen::VectorXi& to) const
	{
		T thetaTo = qs_.calcTheta(to);
		return thetaTo - theta_;
	}

	T logRatio(const JastrowState<T>& rhs) const
	{
		return rhs.theta_ - theta_;
	}

	std::tuple<Eigen::VectorXi, T> data() const
	{
		return std::make_tuple(sigma_, theta_);
	}


	inline int sigmaAt(int i) const
	{
		return sigma_(i);
	}

	inline T getTheta() const
	{
		return theta_;
	}


};

} //namespace nnqs
#endif//CY_NNQS_JASTROW_SAMPLES_HPP
