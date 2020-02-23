#ifndef YANNQ_BASIS_BASJSJZ_HPP
#define YANNQ_BASIS_BASJSJZ_HPP
#include <boost/math/special_functions/binomial.hpp>
//! \ingroup Basis
//! Basis for U(1) symmetric subspace.
class BasisJz
{
private:
	int N_;
	int nup_;
	
public:
	struct BasisJzIterator
	{
		uint32_t n_;
		BasisJzIterator& operator++() //prefix
		{
			next();
			return *this;
		}
		BasisJzIterator operator++(int) //postfix
		{
			BasisJzIterator r(*this);
			next();
			return r;
		}
		void next()
		{
			uint32_t t = n_ | (n_-1);
			uint32_t w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(n_) + 1));
			n_ = w;
		}
		uint32_t operator*() const
		{
			return n_;
		}
		bool operator==(const BasisJzIterator& rhs)
		{
			return n_ == rhs.n_;
		}
		bool operator!=(const BasisJzIterator& rhs)
		{
			return n_ != rhs.n_;
		}
	};

	//! Construct a basis for the subspace. The dimension is \f$N \choose nup\f$.
	//! \param N number of total spins
	//! \param nup number of spin ups(\f$|\uparrow \rangle\f$)
	explicit BasisJz(unsigned int N, unsigned int nup)
		: N_(N), nup_(nup)
	{
	}
	BasisJzIterator begin()
	{
		return BasisJzIterator{(1u<<nup_)-1};
	}
	BasisJzIterator end()
	{
		BasisJzIterator r{((1u<<nup_)-1) << (N_-nup_)};
		r.next();
		return r;
	}
	uint32_t size() const
	{
		uint32_t res = 1;  
	  	uint32_t k = nup_;
		// Since C(n, k) = C(n, n-k)  
		if ( k > N_ - k )  
			k = N_ - k;  
	  
		// Calculate value of  
		// [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]  
		for (uint32_t i = 0u; i < k; ++i)  
		{  
			res *= (N_ - i);  
			res /= (i + 1);  
		}  
    	return res;  
	}
};


#endif//YANNQ_BASIS_BASJSJZ_HPP
