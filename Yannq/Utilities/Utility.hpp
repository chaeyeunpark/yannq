#ifndef CY_NNQS_UTILITY_HPP
#define CY_NNQS_UTILITY_HPP
#include <memory>
#include <random>
#include <utility>

#include "Utilities/type_traits.hpp"

namespace yannq
{
template<typename T>
inline typename std::enable_if<!is_complex_type<T>::value, T>::type logCosh(T x)
{
    const T xp = std::abs(x);
    if (xp <= 12.) {
        return std::log(std::cosh(xp));
    } else {
        const static T log2v = std::log(2.);
        return xp - log2v;
    }   
}
template<typename T>
inline typename std::enable_if<is_complex_type<T>::value, T>::type logCosh(T x)
{
    const auto xr = x.real();
    const auto xi = x.imag();

    T res = logCosh(xr);
    res += std::log(T{std::cos(xi), std::tanh(xr) * std::sin(xi)});

    return res;
}



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

template<class StateT, class AuxData, class Tuple>
inline typename std::enable_if<is_reference_state_type<StateT>::value, StateT>::type 
construct_state(AuxData&& ad, Tuple&& t)
{
	return constructStateT<StateT>(ad, make_rtuple(t), std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type >::value>{});
}

template<class StateT, class AuxData, class Tuple>
inline typename std::enable_if<!is_reference_state_type<StateT>::value, StateT>::type 
construct_state(AuxData&& ad, Tuple&& t)
{
	return constructStateT<StateT>(ad, t, std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}

/*
template<typename StateT, class AuxData, class Container, class Observable>
auto calcObs(const AuxData& ad, const Container& sr, Observable& obs)
	-> typename std::result_of<Observable(StateT)>::type
{
	using ResultType = typename std::result_of<Observable(StateT)>::type;
	ResultType res{};
	for(const auto& elt: sr)
	{
		auto s = construct_state<StateT>(ad, make_rtuple(elt));
		res += obs(s);
	}
	res /= sr.size();
	return res;
}
*/

template <typename RandomEngine>
Eigen::VectorXi randomSigma(int n, RandomEngine& re)
{
	Eigen::VectorXi sigma(n);
	std::uniform_int_distribution<> uid(0, 1);
	//randomly initialize currSigma
	for(int i = 0; i < n; i++)
	{
		sigma(i) = -2*uid(re)+1;
	}
	return sigma;
}
template <typename RandomEngine>
Eigen::VectorXi randomSigma(int n, int nup, RandomEngine& re)
{
	std::vector<int> sigma(n,1);
	//randomly initialize currSigma
	for(int i = nup; i < n; i++)
	{
		sigma[i] = -1;
	}
	std::shuffle(sigma.begin(), sigma.end(), re);
	return Eigen::Map<Eigen::VectorXi>(sigma.data(), n);
}
Eigen::VectorXi toSigma(int length, unsigned long long int val)
{
	Eigen::VectorXi res(length);
	for(int i = 0; i < length; i++)
	{
		res(i) = 1-2*((val >> i) & 1);
	}
	return res;
}

long long int toValue(const Eigen::VectorXi& sigma)
{
	long long int res = 0;
	for(int i = 0; i < sigma.size(); i++)
	{
		res += (1 << i)*((1-sigma(i))/2);
	}
	return res;
}

}//namespace yannq

#endif//CY_NNQS_UTILITY_HPP
