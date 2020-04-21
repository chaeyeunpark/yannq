#ifndef CY_NNQS_TYPE_TRAITS_HPP
#define CY_NNQS_TYPE_TRAITS_HPP

#include <type_traits>
#include <complex>
namespace yannq
{
template<typename T>
struct is_complex_type: public std::false_type {};
template<typename U>
struct is_complex_type<std::complex<U> >: public std::true_type {};


template<typename T> struct remove_complex                   { typedef T type; };
template<typename T> struct remove_complex<std::complex<T> > { typedef T type; };

template<typename T> 
using remove_complex_t = typename remove_complex<T>::type;


template<typename T>
struct is_reference_state_type: public std::false_type {};

template<typename T>
class MachineStateTypes
{
};
template<typename T>
class MachineStateTypesMT
{
};

template<std::size_t N, typename T, typename... types>
struct get_nth_type
{
    using type = typename get_nth_type<N - 1, types...>::type;
};

template<typename T, typename... types>
struct get_nth_type<0, T, types...>
{
    using type = T;
};

template<bool B, typename T = void>
using disable_if = std::enable_if<!B, T>;

#if __cplusplus >= 201703L
// enable only for c++17
template<bool B, typename T = void>
using disable_if_t = std::enable_if_t<!B, T>;
#endif

} //namespace yannq

#endif//CY_NNQS_TYPE_TRAITS_HPP
