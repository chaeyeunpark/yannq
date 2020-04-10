#ifndef YANNQ_MACHINES_RBM_HPP
#define YANNQ_MACHINES_RBM_HPP
#include <random>
#include <bitset>
#include <fstream>
#include <ios>
#include <string>
#include <cassert>
#include <Eigen/Eigen>

#include <tbb/tbb.h>

#include <nlohmann/json.hpp>

#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"
#include "Serializers/SerializeEigen.hpp"

namespace yannq
{
//! \ingroup Machines
//! RBM machine
template<typename T>
class RBM
{
	static_assert(std::is_floating_point<T>::value || is_complex_type<T>::value, "T must be floating or complex");
public:
	using ScalarType = T;
	using RealScalarType = typename yannq::remove_complex<T>::type;

	using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

private:
	uint32_t n_; //# of qubits
	uint32_t m_; //# of hidden units

	bool useBias_;

	MatrixType W_; //W should be m by n
	VectorType a_; //a is length n
	VectorType b_; //b is length m

public:

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "RBM"},
			{"useBias", true},
			{"n", n_},
			{"m", m_}
		};
	}
	inline uint32_t getN() const
	{
		return n_;
	}
	inline uint32_t getM() const
	{
		return m_;
	}

	inline uint32_t getDim() const
	{
		if(useBias_)
			return n_*m_ + n_ + m_;
		else
			return n_*m_;
	}

	inline bool useBias() const
	{
		return useBias_;
	}

	inline VectorType calcTheta(const Eigen::VectorXi& sigma) const
	{
		VectorType s = sigma.cast<T>();
		return W_*s + b_;
	}
	inline VectorType calcGamma(const Eigen::VectorXi& hidden) const
	{
		VectorType h = hidden.cast<T>();
		return W_.transpose()*h + a_;
	}
	
	RBM(uint32_t n, uint32_t m, bool useBias = true)
		: n_(n), m_(m), useBias_(useBias),
		W_(m,n), a_(n), b_(m)
	{
		if(!useBias)
		{
			a_.setZero();
			b_.setZero();
		}
	}

	void setUseBias(bool newBias)
	{
		useBias_ = newBias;
	}

	void resize(uint32_t n, uint32_t m)
	{
		n_ = n;
		m_ = m;

		a_.resize(n);
		b_.resize(m);
		W_.resize(m,n);
	}

	void conservativeResize(uint32_t newM)
	{
		VectorType newB = VectorType::Zero(newM);
		newB.head(m_) = b_;

		MatrixType newW = MatrixType::Zero(newM, n_);
		newW.topLeftCorner(m_, n_) = W_;

		m_ = newM;
		b_ = std::move(newB);
		W_ = std::move(newW);
	}

	template<typename U>
	RBM(const RBM<U>& rhs)
		: n_(rhs.getN()), m_(rhs.getM()), W_(rhs.getW()), a_(rhs.getA()), b_(rhs.getB())
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");
	}

	RBM(const RBM& rhs)
		: n_(rhs.getN()), m_(rhs.getM()), W_(rhs.getW()), a_(rhs.getA()), b_(rhs.getB())
	{
	}
	
	RBM(RBM<T>&& rhs)
		: n_(rhs.n_), m_(rhs.m_), W_(std::move(rhs.W_)), a_(std::move(rhs.a_)), b_(std::move(rhs.b_))
	{
	}

	void setW(const Eigen::Ref<MatrixType>& m)
	{
		assert(m.rows() == W_.rows() && m.cols() == W_.cols());
		W_ = m;
	}

	void setA(const VectorConstRefType& A)
	{
		assert(A.size() == a_.size());
		if(!useBias_)
			return ;
		a_ = A;
	}

	void setB(const VectorConstRefType& B)
	{
		assert(B.size() == b_.size());
		if(!useBias_)
			return ;
		b_ = B;
	}

	template<typename U>
	RBM& operator=(const RBM<U>& rhs)
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");

		if(this == &rhs)
			return *this;

		n_ = rhs.n_;
		m_ = rhs.m_;
		useBias_ = rhs.useBias_;

		W_ = rhs.W_;
		a_ = rhs.a_;
		b_ = rhs.b_;

		return *this;
	}

	~RBM() = default;

	T W(uint32_t j, uint32_t i) const
	{
		return W_.coeff(j,i);
	}
	T A(uint32_t i) const
	{
		return a_.coeff(i);
	}
	T B(uint32_t j) const
	{
		return b_.coeff(j);
	}
	
	const MatrixType& getW() const & { return W_; } 
	MatrixType getW() && { return std::move(W_); } 

	const VectorType& getA() const & { return a_; } 
	VectorType getA() && { return std::move(a_); } 

	const VectorType& getB() const & { return b_; } 
	VectorType getB() && { return std::move(b_); } 


	//! update Bias A by adding v
	void updateA(const VectorConstRefType& v)
	{
		assert(!useBias_);
		a_ += v;
	}
	//! update Bias B by adding v
	void updateB(const VectorConstRefType& v)
	{
		assert(!useBias_);
		b_ += v;
	}
	//! update the weight W by adding m
	void updateW(const Eigen::Ref<const Eigen::MatrixXd>& m)
	{
		W_ += m;
	}

	//! update all parameters.
	void updateParams(const VectorConstRefType& m)
	{
		assert(m.size() == getDim());
		W_ += Eigen::Map<const MatrixType>(m.data(), m_, n_);
		if(!useBias_)
			return ;
		a_ += Eigen::Map<const VectorType>(m.data() + m_*n_, n_);
		b_ += Eigen::Map<const VectorType>(m.data() + m_*n_ + n_, m_);
	}

	VectorType getParams() const
	{
		VectorType res(getDim());
		res.head(n_*m_) = Eigen::Map<const VectorType>(W_.data(), W_.size());
		if(!useBias_)
			return res;

		res.segment(n_*m_, n_) = a_;
		res.segment(n_*m_ + n_, m_) = b_;
		return res;
	}

	void setParams(const VectorConstRefType& r)
	{
		Eigen::Map<VectorType>(W_.data(), W_.size()) = r.head(n_*m_);
		if(!useBias_)
			return ;
		a_ = r.segment(n_*m_, n_);
		b_ = r.segment(n_*m_ + n_, m_);
	}

	bool hasNaN() const
	{
		return a_.hasNaN() || b_.hasNaN() || W_.hasNaN();
	}

	/* When T is real type */
	template <typename RandomEngine, class U=T,
            	std::enable_if_t < !is_complex_type<U>::value, int > = 0 >
	void initializeRandom(RandomEngine& re, T sigma = 1e-3)
	{
		std::normal_distribution<double> nd{0, sigma};
		if(useBias_)
		{
			for(uint32_t i = 0u; i < n_; i++)
			{
				a_.coeffRef(i) = nd(re);
			}
			for(uint32_t i = 0u; i < m_; i++)
			{
				b_.coeffRef(i) = nd(re);
			}
		}
		for(uint32_t j = 0u; j < n_; j++)
		{
			for(uint32_t i = 0u; i < m_; i++)
			{
				W_.coeffRef(i, j) = nd(re);
			}
		}
	}

	/* When T is complex type */
	template <typename RandomEngine, class U=T,
               std::enable_if_t < is_complex_type<U>::value, int > = 0 >
	void initializeRandom(RandomEngine& re, typename remove_complex<T>::type sigma = 1e-3)
	{
		std::normal_distribution<typename remove_complex<T>::type> nd{0, sigma};
		
		if(useBias_)
		{
			for(uint32_t i = 0u; i < n_; i++)
			{
				a_.coeffRef(i) = T{nd(re), nd(re)};
			}
			for(uint32_t i = 0u; i < m_; i++)
			{
				b_.coeffRef(i) = T{nd(re), nd(re)};
			}
		}
		for(uint32_t j = 0; j < n_; j++)
		{
			for(uint32_t i = 0u; i < m_; i++)
			{
				W_.coeffRef(i, j) = T{nd(re), nd(re)};
			}
		}
	}

	bool operator==(const RBM<T>& rhs) const
	{
		if(n_ != rhs.n_ || m_ != rhs.m_)
			return false;
		return (a_ == rhs.a_) || (b_ == rhs.b_) || (W_ == rhs.W_);
	}

	std::tuple<Eigen::VectorXi, VectorType> makeData(const Eigen::VectorXi& sigma) const
	{
		return std::make_tuple(sigma, calcTheta(sigma));
	}

	T logCoeff(const std::tuple<Eigen::VectorXi, VectorType>& t) const
	{
		using std::cosh;

		VectorType ss = std::get<0>(t).template cast<T>();
		T s = a_.transpose()*ss;
		for(uint32_t j = 0u; j < m_; j++)
		{
			s += logCosh(std::get<1>(t).coeff(j));
		}
		return s;
	}

	T coeff(const std::tuple<Eigen::VectorXi, VectorType>& t) const
	{
		using std::cosh;

		VectorType ss = std::get<0>(t).template cast<T>();
		T s = a_.transpose()*ss;
		T p = exp(s) * std::get<1>(t).array().cosh().prod();
		return p;
	}

	VectorType logDeriv(const std::tuple<Eigen::VectorXi, VectorType>& t) const 
	{ 
		VectorType res(getDim()); 

		VectorType tanhs = std::get<1>(t).array().tanh(); 
		VectorType sigma = std::get<0>(t).template cast<ScalarType>();
		
		for(uint32_t i = 0u; i < n_; i++) 
		{ 
			res.segment(i*m_, m_) = sigma(i)*tanhs; 
		}
		if(!useBias_)
			return res;
		res.segment(n_*m_, n_) = sigma;
		res.segment(n_*m_ + n_, m_) = tanhs; 
		return res; 
	} 

	uint32_t widx(uint32_t i, uint32_t j) const
	{
		return i*m_ + j;
	}

};


template<typename T>
typename RBM<T>::VectorType getPsi(const RBM<T>& qs, bool normalize)
{
	const uint32_t n = qs.getN();
	typename RBM<T>::VectorType psi(1u<<n);
	tbb::parallel_for(0u, (1u << n),
		[n, &qs, &psi](uint32_t idx)
	{
		auto s = toSigma(n, idx);
		psi(idx) = qs.coeff(std::make_tuple(s, qs.calcTheta(s)));
	});
	if(normalize)
		psi.normalize();
	return psi;
}

template<typename T, typename Iterable> //Iterable must be random access iterable
typename RBM<T>::VectorType getPsi(const RBM<T>& qs, Iterable&& basis, bool normalize)
{
	const uint32_t n = qs.getN();
	typename RBM<T>::VectorType psi(basis.size());

	tbb::parallel_for(std::size_t(0u), basis.size(),
		[n, &qs, &psi, &basis](std::size_t idx)
	{
		auto s = toSigma(n, basis[idx]);
		psi(idx) = qs.coeff(qs.makeData(s));
	});
	if(normalize)
		psi.normalize();
	return psi;
}
}//namespace yannq

#endif//YANNQ_MACHINES_RBM_HPP
