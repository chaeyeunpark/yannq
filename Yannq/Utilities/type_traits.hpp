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
struct is_reference_state_type: public std::false_type {};

template<typename T>
class MachineStateTypes
{
};
template<typename T>
class MachineStateTypesMT
{
};



} //namespace yannq

#endif//CY_NNQS_TYPE_TRAITS_HPP
