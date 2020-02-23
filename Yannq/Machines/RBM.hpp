#ifndef YANNQ_MACHINES_RBM_HPP
#define YANNQ_MACHINES_RBM_HPP
#include <random>
#include <bitset>
#include <fstream>
#include <ios>
#include <string>
#include <Eigen/Eigen>

#include <cereal/access.hpp> 
#include <cereal/types/memory.hpp>

#include <nlohmann/json.hpp>

#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"
#include "Serializers/SerializeEigen.hpp"

namespace yannq
{
//! \ingroup Machines
//! RBM machine that uses biases
template<typename T, bool useBias = true>
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
	int n_; //# of qubits
	int m_; //# of hidden units

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
	inline int getN() const
	{
		return n_;
	}
	inline int getM() const
	{
		return m_;
	}

	inline int getDim() const
	{
		return n_*m_ + n_ + m_;
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
	
	RBM(int n, int m)
		: n_(n), m_(m), W_(m,n), a_(n), b_(m)
	{
	}

	void resize(int n, int m)
	{
		n_ = n;
		m_ = m;

		a_.resize(n);
		b_.resize(m);
		W_.resize(m,n);
	}

	void conservativeResize(int newM)
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
	RBM(const RBM<U, true>& rhs)
		: n_(rhs.getN()), m_(rhs.getM()), W_(rhs.getW()), a_(rhs.getA()), b_(rhs.getB())
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");
	}

	RBM(const RBM& rhs)
		: n_(rhs.getN()), m_(rhs.getM()), W_(rhs.getW()), a_(rhs.getA()), b_(rhs.getB())
	{
	}
	
	RBM(RBM<T, true>&& rhs)
		: n_(rhs.n_), m_(rhs.m_), W_(std::move(rhs.W_)), a_(std::move(rhs.a_)), b_(std::move(rhs.b_))
	{
	}

	template<typename U>
	RBM(const RBM<U, false>& rhs)
		: n_(rhs.getN()), m_(rhs.getM()), W_(rhs.getW())
	{
		a_.resize(n_);
		b_.resize(m_);
		a_.setZero();
		b_.setZero();
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");
	}
	RBM(RBM<T, false>&& rhs)
		: n_(rhs.n_), m_(rhs.m_), W_(std::move(rhs.W_))
	{
		a_.resize(n_);
		b_.resize(m_);
		a_.setZero();
		b_.setZero();
	}

	void setW(MatrixType m)
	{
		assert(m.rows() == W_.rows() && m.cols() == W_.cols());
		W_ = std::move(m);
	}

	void setA(const VectorConstRefType& A)
	{
		assert(A.size() == a_.size());
		a_ = std::move(A);
	}

	void setB(const VectorConstRefType& B)
	{
		assert(B.size() == b_.size());
		b_ = std::move(B);
	}

	template<typename U>
	RBM& operator=(const RBM<U>& rhs)
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");

		if(this == &rhs)
			return *this;
		n_ = rhs.n_;
		m_ = rhs.m_;

		W_ = rhs.W_;
		a_ = rhs.a_;
		b_ = rhs.b_;

		return *this;
	}

	template<typename U>
	RBM& operator=(RBM<U>&& rhs)
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");

		n_ = rhs.n_;
		m_ = rhs.m_;

		W_ = std::move(rhs.W_);
		a_ = std::move(rhs.a_);
		b_ = std::move(rhs.b_);

		return *this;
	}

	template<typename U>
	RBM& operator=(const RBM<U, false>& rhs)
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");

		if(this == &rhs)
			return *this;
		n_ = rhs.n_;
		m_ = rhs.m_;

		W_ = rhs.W_;
		a_ = VectorType::Zero(n_);
		b_ = VectorType::Zero(m_);

		return *this;
	}

	template<typename U>
	RBM& operator=(RBM<U,false>&& rhs)
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");

		n_ = rhs.n_;
		m_ = rhs.m_;

		W_ = std::move(rhs.W_);
		a_ = VectorType::Zero(n_);
		b_ = VectorType::Zero(m_);

		return *this;
	}

	~RBM() = default;

	T W(int j, int i) const
	{
		return W_.coeff(j,i);
	}
	T A(int i) const
	{
		return a_.coeff(i);
	}
	T B(int j) const
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
		a_ += v;
	}
	//! update Bias B by adding v
	void updateB(const VectorConstRefType& v)
	{
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
		a_ += Eigen::Map<const VectorType>(m.data()+0, n_);
		b_ += Eigen::Map<const VectorType>(m.data()+n_, m_);
		W_ += Eigen::Map<const MatrixType>(m.data()+n_+m_, m_,n_);
	}

	VectorType getParams() const
	{
		VectorType res(getDim());
		res.head(n_) = a_;
		res.segment(n_, m_) = b_;
		res.segment(n_+m_, n_*m_) = Eigen::Map<const VectorType>(W_.data(), W_.size());
		return res;
	}

	void setParams(const VectorConstRefType& r)
	{
		a_ = r.head(n_);
		b_ = r.segment(n_, m_);
		Eigen::Map<VectorType>(W_.data(), W_.size()) = r.segment(n_+m_, n_*m_);
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
		
		for(int i = 0; i < n_; i++)
		{
			a_.coeffRef(i) = nd(re);
		}
		for(int i = 0; i < m_; i++)
		{
			b_.coeffRef(i) = nd(re);
		}
		for(int j = 0; j < n_; j++)
		{
			for(int i = 0; i < m_; i++)
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
		
		for(int i = 0; i < n_; i++)
		{
			a_.coeffRef(i) = T{nd(re), nd(re)};
		}
		for(int i = 0; i < m_; i++)
		{
			b_.coeffRef(i) = T{nd(re), nd(re)};
		}
		for(int j = 0; j < n_; j++)
		{
			for(int i = 0; i < m_; i++)
			{
				W_.coeffRef(i, j) = T{nd(re), nd(re)};
			}
		}
	}

	bool operator==(const RBM<T,useBias>& rhs) const
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
		for(int j = 0; j < m_; j++)
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

		res.head(n_) = std::get<0>(t).template cast<ScalarType>(); 
		res.segment(n_, m_) = tanhs; 

		for(int i = 0; i < n_; i++) 
		{ 
			res.segment(n_+ m_ + i*m_, m_) = res(i)*tanhs; 
		}
		return res; 
	} 

	uint32_t widx(int i, int j) const
	{
		return i*m_ + j;
	}


	/* Serialization using cereal */
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(n_,m_);
		ar(a_, b_, W_);
	}

};

template<typename T>
class RBM<T, false>
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
	int n_; //# of qubits
	int m_; //# of hidden units

	MatrixType W_; //W should be m by n

public:
	inline int getN() const
	{
		return n_;
	}
	inline int getM() const
	{
		return m_;
	}

	inline int getDim() const
	{
		return n_*m_;
	}

	inline VectorType calcTheta(const Eigen::VectorXi& sigma) const
	{
		VectorType s = sigma.cast<T>();
		return W_*s;
	}
	inline VectorType calcGamma(const Eigen::VectorXi& hidden) const
	{
		VectorType h = hidden.cast<T>();
		return W_.transpose()*h;
	}
	
	RBM(int n, int m)
		: n_(n), m_(m), W_(m,n)
	{
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "RBM"},
			{"useBias", false},
			{"n", n_},
			{"m", m_}
		};
	}

	template<typename U>
	RBM(const RBM<U, false>& rhs)
		: n_(rhs.n_), m_(rhs.m_), W_(rhs.W_)
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");
	}
	
	template<typename U>
	RBM(RBM<U, false>&& rhs)
		: n_(rhs.n_), m_(rhs.m_), W_(std::move(rhs.W_))
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");
	}

	VectorType getA() const
	{
		return VectorType::Zero(n_); 
	}

	void setW(MatrixType m)
	{
		assert(m.rows() == W_.rows() && m.cols() == W_.cols());
		W_ = std::move(m);
	}

	template<typename U>
	RBM& operator=(const RBM<U, false>& rhs)
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");

		if(this == &rhs)
			return *this;
		n_ = rhs.n_;
		m_ = rhs.m_;

		W_ = rhs.W_;

		return *this;
	}


	template<typename U>
	RBM& operator=(RBM<U,false>&& rhs)
	{
		static_assert(std::is_convertible<U,T>::value, "U should be convertible to T");

		n_ = rhs.n_;
		m_ = rhs.m_;

		W_ = std::move(rhs.W_);

		return *this;
	}

	~RBM() = default;

	T W(int j, int i) const
	{
		return W_.coeff(j,i);
	}

	//Must be improved..
	T A(int i) const
	{
		(void)i;
		return 0.0;
	}
	T B(int j) const
	{
		(void)j;
		return 0.0;
	}

	void resize(int n, int m)
	{
		n_ = n;
		m_ = m;

		W_.resize(m,n);
	}

	void conservativeResize(int newM)
	{
		MatrixType newW = MatrixType::Zero(newM, n_);
		newW.topLeftCorner(m_, n_) = W_;

		m_ = newM;
		W_ = std::move(newW);
	}

	const MatrixType& getW() const & { return W_; } 
	MatrixType getW() && { return std::move(W_); } 

	//! update the weight W by adding m
	void updateW(const Eigen::Ref<const Eigen::MatrixXd>& m)
	{
		W_ += m;
	}
	
	//! update the weight
	void updateParams(const VectorConstRefType& m)
	{
		assert(m.size() == getDim());
		W_ += Eigen::Map<const MatrixType>(m.data(), m_,n_);
	}

	VectorType getParams() const
	{
		return Eigen::Map<VectorType>(W_.data(), W_.size());
	}

	void setParams(const VectorConstRefType& r)
	{
		Eigen::Map<VectorType>(W_.data(), W_.size()) = r;
	}


	bool hasNaN() const
	{
		return W_.hasNaN();
	}

	/* When T is real type */
	template <typename RandomEngine, class U=T,
               std::enable_if_t < !is_complex_type<U>::value, int > = 0 >
	void initializeRandom(RandomEngine& re, T sigma = 0.001)
	{
		std::normal_distribution<double> nd{0, sigma};
		for(int j = 0; j < n_; j++)
		{
			for(int i = 0; i < m_; i++)
			{
				W_.coeffRef(i, j) = nd(re);
			}
		}
	}

	/* When T is complex type */
	template <typename RandomEngine, class U=T,
               std::enable_if_t < is_complex_type<U>::value, int > = 0 >
	void initializeRandom(RandomEngine& re, typename remove_complex<T>::type weight = 1e-3)
	{
		std::uniform_real_distribution<typename remove_complex<T>::type> nd{-0.5*weight,0.5*weight};
		
		for(int j = 0; j < n_; j++)
		{
			for(int i = 0; i < m_; i++)
			{
				W_.coeffRef(i, j) = T{nd(re), nd(re)};
			}
		}
	}

	bool operator==(const RBM<T, false>& rhs) const
	{
		if(n_ != rhs.n_ || m_ != rhs.m_)
			return false;
		return (W_ == rhs.W_);
	}
	
	std::tuple<Eigen::VectorXi, VectorType> makeData(const Eigen::VectorXi& sigma) const
	{
		return std::make_tuple(sigma, calcTheta(sigma));
	}

	T logCoeff(const std::tuple<Eigen::VectorXi, VectorType>& t) const
	{
		using std::cosh;
		T s{};
		for(int j = 0; j < m_; j++)
		{
			s += logCosh(std::get<1>(t).coeff(j));
		}
		return s;
	}

	T coeff(const std::tuple<Eigen::VectorXi, VectorType>& t) const
	{
		using std::cosh;

		return std::get<1>(t).array().cosh().prod();
	}
	
	/*
	VectorType logDeriv(const std::tuple<Eigen::VectorXi, VectorType>& t) const
	{
		VectorType res(getDim());
		for(int i = 0; i < n_; i++)
		{
			for(int j = 0; j < m_; j++)
			{
				res(i*m_ + j) = T(std::get<0>(t)(i))*tanh(std::get<1>(t)(j));
			}
		}
		return res;
	}
	*/
	VectorType logDeriv(const std::tuple<Eigen::VectorXi, VectorType>& t) const 
	{ 
		VectorType res(getDim()); 

		VectorType tanhs = std::get<1>(t).array().tanh(); 

		for(int i = 0; i < n_; i++) 
		{ 
			res.segment(i*m_, m_) = res(i)*tanhs; 
		}
		return res; 
	} 

	/* Serialization using cereal */
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(n_,m_);
		ar(W_);
	}
};

template<typename T, bool useBias>
typename RBM<T, useBias>::VectorType getPsi(const RBM<T, useBias>& qs, bool normalize)
{
	const int n = qs.getN();
	typename RBM<T>::VectorType psi(1<<n);
#pragma omp parallel for schedule(static,8)
	for(uint64_t i = 0; i < (1u<<n); i++)
	{
		auto s = toSigma(n, i);
		psi(i) = qs.coeff(std::make_tuple(s, qs.calcTheta(s)));
	}
	if(normalize)
		return psi.normalized();
	else
		return psi;
}

template<typename T, bool useBias>
typename RBM<T, useBias>::VectorType getPsi(const RBM<T, useBias>& qs, const std::vector<uint32_t>& basis, bool normalize)
{
	const int n = qs.getN();
	typename RBM<T>::VectorType psi(basis.size());
#pragma omp parallel for schedule(static,8)
	for(uint64_t i = 0; i < basis.size(); i++)
	{
		auto s = toSigma(n, basis[i]);
		psi(i) = qs.coeff(std::make_tuple(s, qs.calcTheta(s)));
	}
	if(normalize)
		return psi.normalized();
	else
		return psi;
}

template<typename T, bool useBias, class BasisIter>
typename RBM<T, useBias>::RealScalarType getNorm(const RBM<T, useBias>& qs, BasisIter&& basis)
{
	const int n = qs.getN();
	typename RBM<T>::VectorType psi(basis.size());
	for(auto sigma: basis)
	{
		auto s = toSigma(n, sigma);
		psi(sigma) = qs.coeff(std::make_tuple(s, qs.calcTheta(s)));
	}
	return psi.squaredNorm();

}
}//namespace yannq

namespace cereal
{
	template <typename T>
	struct LoadAndConstruct<yannq::RBM<T, false> >
	{
		template<class Archive>
		static void load_and_construct(Archive& ar, cereal::construct<yannq::RBM<T,false>>& construct)
		{
			int n,m;
			ar(n,m);
			construct(n,m);
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W;
			ar(W);
			construct->setW(W);
		}
	};
	template <typename T>
	struct LoadAndConstruct<yannq::RBM<T, true> >
	{
		template<class Archive>
		static void load_and_construct(Archive& ar, cereal::construct<yannq::RBM<T,true> >& construct)
		{
			int n,m;
			ar(n,m);
			construct(n,m);
			Eigen::Matrix<T, Eigen::Dynamic, 1> A;
			Eigen::Matrix<T, Eigen::Dynamic, 1> B;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W;
			ar(A,B,W);
			construct->setA(A);
			construct->setB(B);
			construct->setW(W);
		}
	};
}//namespace cereal
#endif//YANNQ_MACHINES_RBM_HPP
