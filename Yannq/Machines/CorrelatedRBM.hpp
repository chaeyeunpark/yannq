#pragma once
#include <random>
#include <bitset>
#include <fstream>
#include <ios>
#include <string>
#include <cassert>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

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
class CorrelatedRBM
{
	static_assert(std::is_floating_point<T>::value || is_complex_type<T>::value, "T must be floating or complex");
public:
	using Scalar = T;
	using RealScalar = typename yannq::remove_complex<T>::type;

	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	using RealVector = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
	
	using DataT = std::tuple<Eigen::VectorXi, Vector>;

private:
	uint32_t n_; //# of qubits
	uint32_t m_; //# of hidden units

	bool useBias_;

	Eigen::SparseMatrix<int> correl_; 


	Matrix W_; //W should be m by (n + k)
	Vector a_; //a is length n + k
	Vector b_; //b is length m

public:
	/* \param n number of qubits
	 * \param m number of hidden units
	 * \param correl n \times k matrix. For each column, 1 is given at the location of
	 * a qubit involving a correlation. C_k = \prod_{l, correl(l,k)=1} s_l
	 * */
	CorrelatedRBM(uint32_t n, uint32_t m, 
			const Eigen::SparseMatrix<int>& correl, bool useBias = true) noexcept
		: n_(n), m_(m), useBias_(useBias), correl_{correl}, 
		W_(m,n+correl.cols()), a_(n+correl.cols()), b_(m) 
	{
		correl_.makeCompressed();
		a_.setZero();
		b_.setZero();
		W_.setZero();
	}

	CorrelatedRBM() noexcept
		: useBias_{true}
	{
	}

	template<typename U, std::enable_if_t<std::is_convertible_v<U, T> && !std::is_same_v<U, T>, int> = 0>
	CorrelatedRBM(const CorrelatedRBM<U>& rhs) 
		: n_{rhs.getN()}, m_{rhs.getM()}, useBias_{rhs.useBias()}, correl_{rhs.correl_}
	{
		W_ = rhs.getW().template cast<T>();
		a_ = rhs.getA().template cast<T>();
		b_ = rhs.getB().template cast<T>();
	}

	CorrelatedRBM(const CorrelatedRBM& rhs) /* noexcept */ = default;
	CorrelatedRBM(CorrelatedRBM&& rhs) /* noexcept */ = default;

	template<typename U, std::enable_if_t<std::is_convertible_v<U, T> && !std::is_same_v<U, T>, int> = 0>
	CorrelatedRBM& operator=(const CorrelatedRBM<U>& rhs) 
	{
		n_ = rhs.n_;
		m_ = rhs.m_;
		useBias_ = rhs.useBias_;
		correl_ = rhs.correl_;

		W_ = rhs.W_.template cast<T>();
		a_ = rhs.a_.template cast<T>();
		b_ = rhs.b_.template cast<T>();

		return *this;
	}

	CorrelatedRBM& operator=(const CorrelatedRBM& rhs) /* noexcept */ = default;
	CorrelatedRBM& operator=(CorrelatedRBM&& rhs) /* noexcept */ = default;

	template<typename U, std::enable_if_t<std::is_convertible_v<T, U>, int> = 0>
	CorrelatedRBM<U> cast() const
	{
		CorrelatedRBM<U> res(n_, m_, correl_, useBias_);
		res.setA(a_.template cast<U>());
		res.setB(b_.template cast<U>());
		res.setW(W_.template cast<U>());
		return res;
	}

	nlohmann::json desc() const
	{
		return nlohmann::json
		{
			{"name", "CorrelatedRBM"},
			{"useBias", useBias_},
			{"k", correl_.rows()},
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

	inline uint32_t getK() const
	{
		return correl_.cols();
	}

	inline uint32_t getDim() const
	{
		auto k = getK();
		if(useBias_)
			return (n_ + k)*m_ + n_ + k + m_;
		else
			return (n_ + k)*m_;
	}


	inline bool useBias() const
	{
		return useBias_;
	}

	Eigen::VectorXi calcCorrel(const Eigen::VectorXi& sigma) const
	{
		Eigen::VectorXi c = Eigen::VectorXi::Ones(getK());
		for(uint32_t l = 0; l < getK(); ++l) //cols (outer)
		{
			for(Eigen::SparseMatrix<int>::InnerIterator it(correl_, l); it; ++it)
			{
				c(l) *= sigma(it.row());
			}
		}
		return c;
	}

	/**
	 * \param ss is sigma plus correl
	 */
	inline Vector calcTheta(const Eigen::VectorXi& ss) const
	{
		assert(ss.size() == n_ + getK());
		Vector s = ss.cast<T>();
		return W_*s + b_;
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
		b_.resize(m + getK());
		W_.resize(m + getK(),n);

		if(!useBias_)
		{
			a_.setZero();
			b_.setZero();
		}
	}

	void conservativeResize(uint32_t newM)
	{
		Vector newB = Vector::Zero(newM);
		newB.head(m_) = b_;

		Matrix newW = Matrix::Zero(newM, n_ + getK());
		newW.topRows(m_) = W_;

		m_ = newM;
		b_ = std::move(newB);
		W_ = std::move(newW);
	}

	void setCorrel(const Eigen::SparseMatrix<int>& correl)
	{
		correl_ = correl;
	}

	void setW(const Eigen::Ref<const Matrix>& m)
	{
		assert(m.rows() == W_.rows() && m.cols() == W_.cols());
		W_ = m;
	}

	void setA(const VectorConstRef& A)
	{
		assert(A.size() == a_.size());
		if(!useBias_)
			return ;
		a_ = A;
	}

	void setB(const VectorConstRef& B)
	{
		assert(B.size() == b_.size());
		if(!useBias_)
			return ;
		b_ = B;
	}


	inline const T& W(uint32_t j, uint32_t i) const
	{
		return W_.coeff(j,i);
	}
	inline const T& A(uint32_t i) const
	{
		return a_.coeff(i);
	}
	inline const T& B(uint32_t j) const
	{
		return b_.coeff(j);
	}

	inline T& W(uint32_t j, uint32_t i) 
	{
		return W_.coeffRef(j,i);
	}
	inline T& A(uint32_t i) 
	{
		return a_.coeffRef(i);
	}
	inline T& B(uint32_t j) 
	{
		return b_.coeffRef(j);
	}

	Eigen::SparseMatrix<int> getCorrel() const
	{
		return correl_;
	}

	
	const Matrix& getW() const & { return W_; } 
	Matrix getW() && { return std::move(W_); } 

	const Vector& getA() const & { return a_; } 
	Vector getA() && { return std::move(a_); } 

	const Vector& getB() const & { return b_; } 
	Vector getB() && { return std::move(b_); } 


	//! update Bias A by adding v
	void updateA(const VectorConstRef& v)
	{
		assert(useBias_);
		a_ += v;
	}
	//! update Bias B by adding v
	void updateB(const VectorConstRef& v)
	{
		assert(useBias_);
		b_ += v;
	}
	//! update the weight W by adding m
	void updateW(const Eigen::Ref<const Matrix>& m)
	{
		W_ += m;
	}

	//! update all parameters.
	void updateParams(const VectorConstRef& m)
	{
		assert(m.size() == getDim());
		uint32_t k = getK();
		W_ += Eigen::Map<const Matrix>(m.data(), m_, n_ + k);
		if(!useBias_)
			return ;
		a_ += Eigen::Map<const Vector>(m.data() + m_*(n_ + k), n_ + k);
		b_ += Eigen::Map<const Vector>(m.data() + m_*(n_ + k) + n_ + k, m_);
	}

	Vector getParams() const
	{
		const auto k = getK();
		Vector res(getDim());
		res.head((n_+k)*m_) = Eigen::Map<const Vector>(W_.data(), W_.size());
		if(!useBias_)
			return res;

		res.segment((n_ + k)*m_, n_ + k) = a_;
		res.segment((n_ + k)*m_ + n_ + k, m_) = b_;
		return res;
	}

	void setParams(const VectorConstRef& r)
	{
		assert(r.size() == getDim());
		const auto k = getK();
		Eigen::Map<Vector>(W_.data(), W_.size()) = r.head((n_ + k)*m_);
		if(!useBias_)
			return ;
		a_ = r.segment((n_+k)*m_, n_ + k);
		b_ = r.segment((n_+k)*m_ + n_ + k, m_);
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
		const auto k = getK();
		std::normal_distribution<T> nd{0, sigma};
		if(useBias_)
		{
			for(uint32_t i = 0u; i < n_ + k; i++)
			{
				a_.coeffRef(i) = nd(re);
			}
			for(uint32_t i = 0u; i < m_; i++)
			{
				b_.coeffRef(i) = nd(re);
			}
		}
		for(uint32_t j = 0u; j < n_ + k; j++)
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
		const auto k = getK();
		std::normal_distribution<typename remove_complex<T>::type> nd{0, sigma};
		
		if(useBias_)
		{
			for(uint32_t i = 0u; i < n_ + k; i++)
			{
				a_.coeffRef(i) = T{nd(re), nd(re)};
			}
			for(uint32_t i = 0u; i < m_; i++)
			{
				b_.coeffRef(i) = T{nd(re), nd(re)};
			}
		}
		for(uint32_t j = 0; j < n_ + k; j++)
		{
			for(uint32_t i = 0u; i < m_; i++)
			{
				W_.coeffRef(i, j) = T{nd(re), nd(re)};
			}
		}
	}

	bool operator==(const CorrelatedRBM<T>& rhs) const
	{
		if(n_ != rhs.n_ || m_ != rhs.m_ || getK() != rhs.getK())
			return false;
		if((correl_ - rhs.corre_).norm() > 1e-6)
			return false;
		if(useBias_)
			return (a_ == rhs.a_) && (b_ == rhs.b_) && (W_ == rhs.W_);
		else
			return (W_ == rhs.W_);
	}

	std::tuple<Eigen::VectorXi, Vector> makeData(const Eigen::VectorXi& sigma) const
	{
		Eigen::VectorXi ss(n_ + getK());
		ss.head(n_) = sigma;
		ss.tail(getK()) = calcCorrel(sigma);
		return std::make_tuple(ss, calcTheta(ss));
	}

	T logCoeff(const std::tuple<Eigen::VectorXi, Vector>& t) const
	{
		using std::cosh;

		Vector ss = std::get<0>(t).template cast<T>();
		T s = a_.transpose()*ss;
		for(uint32_t j = 0u; j < m_; j++)
		{
			s += logCosh(std::get<1>(t).coeff(j));
		}
		return s;
	}

	T coeff(const std::tuple<Eigen::VectorXi, Vector>& t) const
	{
		using std::cosh;

		Vector ss = std::get<0>(t).template cast<T>();
		T s = a_.transpose()*ss;
		T p = exp(s) * std::get<1>(t).array().cosh().prod();
		return p;
	}

	Vector logDeriv(const std::tuple<Eigen::VectorXi, Vector>& t) const 
	{ 
		Vector res(getDim()); 
		const auto k = getK();

		Vector tanhs = std::get<1>(t).array().tanh(); 
		Vector ss = std::get<0>(t).template cast<Scalar>();
		
		for(uint32_t i = 0u; i < n_ + k; i++) 
		{ 
			res.segment(i*m_, m_) = ss(i)*tanhs; 
		}
		if(!useBias_)
			return res;
		res.segment((n_+k)*m_, n_ + k) = ss;
		res.segment((n_+k)*m_ + n_ + k, m_) = tanhs; 
		return res; 
	} 
};


template<typename T>
typename CorrelatedRBM<T>::Vector getPsi(const CorrelatedRBM<T>& qs, bool normalize)
{
	const uint32_t n = qs.getN();
	typename CorrelatedRBM<T>::Vector psi(1u<<n);
	tbb::parallel_for(0u, (1u << n),
		[n, &qs, &psi](uint32_t idx)
	{
		auto s = toSigma(n, idx);
		psi(idx) = qs.coeff(qs.makeData(s));
	});
	if(normalize)
		psi.normalize();
	return psi;
}

template<typename T, typename Iterable> //Iterable must be random access iterable
typename CorrelatedRBM<T>::Vector getPsi(const CorrelatedRBM<T>& qs, Iterable&& basis, bool normalize)
{
	const uint32_t n = qs.getN();
	typename CorrelatedRBM<T>::Vector psi(basis.size());

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
