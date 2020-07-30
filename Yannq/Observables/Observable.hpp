#pragma once

#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

namespace yannq
{
template<class Derived>
class Observable
{
public:
	void initIter(uint32_t nsmp)
	{
		static_cast<Derived*>(this)->initIter(nsmp);
	}

	template<class State>
	void eachSample(uint32_t idx, State&& state)
	{
		static_cast<Derived*>(this)->eachSample(idx, std::forward<State>(state));
	}

	void finIter()
	{
		static_cast<Derived*>(this)->finIter();
	}

	template<typename MatrixT>
	void finIter(const Eigen::MatrixBase<MatrixT>& weights)
	{
		static_cast<Derived*>(this)->finIter(weights);
	}

};
}// namespace yannq
