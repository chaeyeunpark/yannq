#ifndef CY_NNQS_JASTROW_HPP
#define CY_NNQS_JASTROW_HPP
#include <random>
#include <Eigen/Dense>
#include "Utilities/type_traits.hpp"
#include <nlohmann/json.hpp>
namespace yannq
{
template<typename T>
class Jastrow
{

public:
	using ScalarType=T;
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<T, Eigen::Dynamic, 1>  Vector;

private:
	int n_;

	Vector a_;
	Matrix J_;

public:

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "Jastrow"},
			{"n", n_},
		};
	}


	inline int getN() const
	{
		return n_;
	}

	Jastrow(int n)
		: n_(n), a_(n_), J_(n,n)
	{
	}

	void resize(int n)
	{
		n_ = n;
		a_.resize(n);
		J_.resize(n,n);
	}

	int getDim() const
	{
		return n_ + n_*(n_-1)/2;
	}

	Vector getA() const
	{
		return a_;
	}
	Matrix getJ() const
	{
		return J_;
	}

	void setA(const Vector& a) const
	{
		a_ = a;
	}

	void setJ(const Matrix& J) const
	{
		J_ = J;
	}



	T A(int i) const
	{
		return a_(i);
	}
	T J(int i, int j) const
	{
		return J_(i,j);
	}

	template <typename RandomEngine, class U=T,
               typename std::enable_if < !is_complex_type<U>::value, int >::type = 0 >
	void initializeRandom(RandomEngine& re, T weight = 0.001)
	{
		std::normal_distribution<double> nd{};
		
		for(int i = 0; i < n_; i++)
		{
			a_(i) = weight*nd(re);
		}
		J_.setZero();
		for(int j = n_-1; j >=0; --j)
		{
			for(int i = 0; i < j; i++)
			{
				J_(i, j) = weight*nd(re);
			}
		}
	}
	template <typename RandomEngine, class U=T,
               typename std::enable_if < is_complex_type<U>::value, int >::type = 0 >
	void initializeRandom(RandomEngine& re, T weight = 0.001)
	{
		std::normal_distribution<double> nd{};
		
		for(int i = 0; i < n_; i++)
		{
			a_(i) = weight*T(nd(re),nd(re));
		}
		J_.setZero();
		for(int j = n_-1; j >=0; --j)
		{
			for(int i = 0; i < j; i++)
			{
				J_(i, j) = weight*T(nd(re),nd(re));
			}
		}
	}

	T calcTheta(const Eigen::VectorXi& sigma) const
	{
		Vector s = sigma.cast<T>();
		T res = a_.transpose()*s;
		res += T(s.transpose()*J_*s);
		return res;
	}

	std::tuple<Eigen::VectorXi, T> makeData(const Eigen::VectorXi& sigma) const
	{
		return std::make_tuple(sigma, calcTheta(sigma));
	}

	Vector logDeriv(const std::tuple<Eigen::VectorXi, T>& t) const
	{
		Vector res(getDim());
		for(int i = 0; i < n_; i++)
		{
			res(i) = std::get<0>(t)(i);
		}
		for(int i = 0; i < n_; i++)
		{
			for(int j = i+1; i < n_; j++)
			{
				res(i*n_+j+n_) = std::get<0>(t)(i)*std::get<0>(t)(j);
			}
		}
		return res;
	}

	void updateParams(const Vector& u)
	{
		a_ += u.segment(0, n_);
		for(int i = 0; i < n_; i++)
		{
			for(int j = i+1; i < n_; j++)
			{
				J_(i, j) += u(i*n_ + j + n_);
			}
		}
	}


};
}
#endif//CY_NNQS_JASTROW_HPP
