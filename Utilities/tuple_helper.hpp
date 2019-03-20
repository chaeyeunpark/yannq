#ifndef CY_TUPLE_HELPER_HPP
#define CY_TUPLE_HELPER_HPP
namespace nnqs
{
//Chromium implementaion of index sequence template
//
//
//
// Represents a compile-time sequence of integers.
// T is the integer type of the sequence, can be size_t, int.
// I is a parameter pack representing the sequence.
template <class T, T... I> struct integer_sequence {
 typedef T value_type;

 static constexpr size_t size() { return sizeof...(I); }
};

// Alias for the common case of a sequence of size_t.
// A pre-defined sequence for common use case.
template <std::size_t... I>
struct index_sequence : integer_sequence<std::size_t, I...> {};

template <std::size_t N, std::size_t... I>
struct build_index_impl : build_index_impl<N - 1, N - 1, I...> {};
template <std::size_t... I>
struct build_index_impl<0, I...> : index_sequence<I...> {};

// Creates a compile-time integer sequence for a parameter pack.
template <class... Ts>
struct index_sequence_for : build_index_impl<sizeof...(Ts)> {};
}
#endif//CY_TUPLE_HELPER_HPP
