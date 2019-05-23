#ifndef YANNQ_BASIS_BASJSJZ_HPP
#define YANNQ_BASIS_BASJSJZ_HPP

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
	BasisJz(int N, int nup)
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
};


#endif//YANNQ_BASIS_BASJSJZ_HPP
