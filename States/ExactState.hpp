#ifndef CY_NNQS_EXACTSTATE_HPP
#define CY_NNQS_EXACTSTATE_HPP
#include "Utilities/type_traits.hpp"

namespace nnqs
{

class ExactState
{
private:
	const Eigen::VectorXcd& psi_;
	const unsigned long u_;

public:
	using T = ScalarT;

	ExactState(const Eigen::VectorXcd& psi, const unsigned long u)
		: psi_(psi), u_(u)
	{
	}
	T logRatio(int k) const
	{
		return res = std::log(psi_(u_ ^ (1u<<k))) - std::log(psi_(u_));
	}

	T logRatio(int k, int l) const
	{
		return res = std::log(psi_(u_ ^ (1u << k) ^ (1u << l))) - std::log(psi_(u_));
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
		u ^= (1u << k);
	}

	void flip(int k, int l)
	{
		u ^= (1u << k);
		u ^= (1u << l);
	}

	template<std::size_t N>
	void flip(const std::array<int, N>& v)
	{
		for(auto a : v)
		{
			u ^= (1u << a);
		}
	}
	std::tuple<unsigned long> data() const
	{
		return std::make_tuple(u_);
	}

	inline int sigmaAt(int i) const
	{
		return (1 - 2*((u >> i) & 1));
	}



};

} //namespace nnqs
#endif//CY_NNQS_EXACTSTATE_HPP
