#ifndef YANNQ_BASIS_BASISFULL_HPP
#define YANNQ_BASIS_BASISFULL_HPP
#include <cstdint>
class BasisFull
{
private:
	int N_;
public:
	struct BasisFullIterator
	{
		uint32_t n_;
		BasisFullIterator& operator++() //prefix
		{
			n_++;
			return *this;
		}
		BasisFullIterator operator++(int) //postfix
		{
			BasisFullIterator r(*this);
			++n_;
			return r;
		}
		uint32_t operator*() const
		{
			return n_;
		}
		bool operator==(const BasisFullIterator& rhs)
		{
			return n_ == rhs.n_;
		}
		bool operator!=(const BasisFullIterator& rhs)
		{
			return n_ != rhs.n_;
		}
	};
	BasisFull(int N)
		: N_(N)
	{
	}
	BasisFullIterator begin()
	{
		return BasisFullIterator{0};
	}
	BasisFullIterator end()
	{
		return BasisFullIterator{(1u<<N_)};
	}
};

#endif//YANNQ_BASIS_BASISFULL_HPP
