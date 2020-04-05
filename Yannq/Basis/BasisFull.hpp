#ifndef YANNQ_BASIS_BASISFULL_HPP
#define YANNQ_BASIS_BASISFULL_HPP
#include <cstdint>
#include <iterator>

//! \ingroup Basis
//! Basis set for the full Hilbert space in the computational basis.

namespace yannq
{
struct BasisFullIterator;
}

template<>
struct std::iterator_traits<yannq::BasisFullIterator>
{
	using difference_type = int32_t;
	using value_type = uint32_t;
	using pointer = void;
	using reference = uint32_t&;
	using iterator_category = std::random_access_iterator_tag;
};

namespace yannq
{
struct BasisFullIterator
{
	uint32_t n_;
	explicit BasisFullIterator(uint32_t n)
		: n_{n}
	{
	}
	explicit operator uint32_t() const
	{
		return n_;
	}
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
	BasisFullIterator& operator+=(int32_t rhs)
	{
		n_ += rhs;
		return *this;
	}
	BasisFullIterator& operator-=(int32_t rhs)
	{
		n_ -= rhs;
		return *this;
	}
	uint32_t operator[](uint32_t idx)
	{
		return idx;
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

BasisFullIterator operator+(const BasisFullIterator& lhs, int32_t rhs)
{
	auto r = BasisFullIterator{lhs.n_};
	r.n_ += rhs;
	return r;
}

BasisFullIterator operator+(int32_t lhs, const BasisFullIterator& rhs)
{
	auto r = BasisFullIterator{rhs.n_};
	r.n_ += lhs;
	return r;
}

BasisFullIterator operator-(const BasisFullIterator& lhs, int32_t rhs)
{
	auto r = BasisFullIterator{lhs.n_};
	r.n_ -= rhs;
	return r;
}

uint32_t operator-(const BasisFullIterator& lhs, const BasisFullIterator& rhs)
{
	return lhs.n_ - rhs.n_;
}

bool operator<(const BasisFullIterator& lhs, const BasisFullIterator& rhs)
{
	return (lhs.n_ < rhs.n_);
}

bool operator>(const BasisFullIterator& lhs, const BasisFullIterator& rhs)
{
	return (lhs.n_ > rhs.n_);
}

bool operator<=(const BasisFullIterator& lhs, const BasisFullIterator& rhs)
{
	return (lhs.n_ <= rhs.n_);
}

bool operator>=(const BasisFullIterator& lhs, const BasisFullIterator& rhs)
{
	return (lhs.n_ >= rhs.n_);
}




class BasisFull
{
private:
	int N_;
public:
	
	explicit BasisFull(int N)
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
	uint32_t size() const
	{
		return (1u<<N_);
	}
};
}//namespace yannq
#endif//YANNQ_BASIS_BASISFULL_HPP
