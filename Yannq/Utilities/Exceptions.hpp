#ifndef YANNQ_UTILITIES_EXCEPTIONS_HPP
#define YANNQ_UTILITIES_EXCEPTIONS_HPP
#include <exception>
#include <string>
namespace yannq
{
class InvalidArgument : public std::exception
{
private:
	std::string msg_;
public:
	InvalidArgument(const char* msg)
		: msg_{msg}
	{
	}
	virtual const char* what() const noexcept
	{
		return msg_.c_str();
	}
};
}
#endif//YANNQ_UTILITIES_EXCEPTIONS_HPP
