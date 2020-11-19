#pragma once
#include <tuple>
#include "./RBMState.hpp"

namespace yannq
{

namespace detail
{
template<typename ...T, size_t... I>
auto make_rtuple_helper(const std::tuple<T...>& t ,  std::index_sequence<I...>)
-> std::tuple<const T&...>
{ return std::tie(std::get<I>(t)...) ;}

template<typename ...T>
std::tuple<const T&...> make_rtuple(const std::tuple<T...>& t )
{
    return make_rtuple_helper( t, std::index_sequence_for<T...>{});
}

template<class StateT, class AuxData, class Tuple, size_t... Is>
StateT constructStateT(AuxData&& ad, Tuple&& t, std::index_sequence<Is...>)
{
	return StateT{ad, std::get<Is>(t)...};
}
} //namespace yannq::detail

template<class StateT, class AuxData, class Tuple>
inline typename std::enable_if<is_reference_state_type<StateT>::value, StateT>::type 
construct_state(AuxData&& ad, Tuple&& t)
{
	return detail::constructStateT<StateT>(ad, detail::make_rtuple(t), std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type >::value>{});
}

template<class StateT, class AuxData, class Tuple>
inline typename std::enable_if<!is_reference_state_type<StateT>::value, StateT>::type 
construct_state(AuxData&& ad, Tuple&& t)
{
	return detail::constructStateT<StateT>(ad, t, std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}



} //namespace yannq
