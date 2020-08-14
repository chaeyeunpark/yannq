#pragma once
#include <tbb/tbb.h>
#include "States/utils.hpp"
namespace yannq
{
template<typename StateValue, typename Container>
tbb::concurrent_vector<StateValue> dataToState(const typename StateValue::Machine& qs, Container&& container)
{
	using Machine = typename StateValue::Machine;
	tbb::concurrent_vector<StateValue> res;
	tbb::parallel_for_each(begin(container), end(container), 
		[&](const typename Machine::DataT& data)
		{
			res.push_back(construct_state<StateValue>(qs, data));
		}
	);
	return res;
}

} //namespace yannq
