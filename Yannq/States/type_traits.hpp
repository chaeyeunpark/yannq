#pragma once
#include <type_traits>
namespace yannq
{

template<typename T>
struct is_reference_state_type: public std::false_type {};

template<typename T>
class MachineStateTypes
{
};


}
