#ifndef CY_NNQS_UTILITY_HPP
#define CY_NNQS_UTILITY_HPP
#include <memory>
#include <random>
#include <utility>

#include <Eigen/Dense>
#include "Utilities/type_traits.hpp"

namespace yannq
{

//! stable implementation of log(cosh(x)) for real x
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

//! stable implementation of log(cosh(x)) for complex x
template<typename T>
inline typename std::enable_if<is_complex_type<T>::value, T>::type logCosh(T x)
{
    const auto xr = x.real();
    const auto xi = x.imag();

    T res = logCosh(xr);
    res += std::log(T{std::cos(xi), std::tanh(xr) * std::sin(xi)});

    return res;
}

template<typename T>
typename std::enable_if<!is_complex_type<T>::value, T>::type real(const T& v)
{
	return v;
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

//! generate random vector in the computational basis
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
//! generate random vector in the computational basis with the constraint that
//! the number of spin ups is nup.
template <typename RandomEngine>
Eigen::VectorXi randomSigma(int n, int nup, RandomEngine& re)
{
	std::vector<int> sigma(n,-1);
	//randomly initialize currSigma
	for(int i = nup; i < n; i++)
	{
		sigma[i] = 1;
	}
	std::shuffle(sigma.begin(), sigma.end(), re);
	return Eigen::Map<Eigen::VectorXi>(sigma.data(), n);
}

//! generate a vector that the binary representation is val.
Eigen::VectorXi toSigma(int length, uint32_t val)
{
	Eigen::VectorXi res(length);
	for(int i = 0; i < length; i++)
	{
		res(i) = 1-2*((val >> i) & 1);
	}
	return res;
}

//! returns the binary representation of sigma.
long long int toValue(const Eigen::VectorXi& sigma)
{
	long long int res = 0;
	for(int i = 0; i < sigma.size(); i++)
	{
		res += (1 << i)*((1-sigma(i))/2);
	}
	return res;
}


//! for complex type T, generate a vector filled with samples from normal distribution.
template<typename T, class RandomEngine, typename std::enable_if<is_complex_type<T>::value, int>::type = 0 >
Eigen::Matrix<T, Eigen::Dynamic, 1> randomVector(RandomEngine&& re, 
		remove_complex_t<T> sigma, std::size_t nelt)
{
	Eigen::Matrix<T, Eigen::Dynamic, 1> res(nelt);
	std::normal_distribution<typename remove_complex<T>::type> dist(0.0, sigma);
	for(std::size_t i = 0; i < nelt; i++)
	{
		res(i) = T{dist(re),dist(re)};
	}
	return res;
}
//! for real type T, generate a vector filled with samples from normal distribution.
template<typename T, class RandomEngine, typename std::enable_if<!is_complex_type<T>::value, int>::type = 0 >
Eigen::Matrix<T, Eigen::Dynamic, 1> randomVector(RandomEngine&& re, 
		remove_complex_t<T> sigma, std::size_t nelt)
{
	Eigen::Matrix<T, Eigen::Dynamic, 1> res(nelt);
	std::normal_distribution<T> dist(0.0, sigma);
	for(std::size_t i = 0; i < nelt; i++)
	{
		res(i) = dist(re);
	}
	return res;
}

}//namespace yannq

#endif//CY_NNQS_UTILITY_HPP
