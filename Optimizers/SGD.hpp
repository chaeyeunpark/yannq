#ifndef CY_SGD_HPP
#define CY_SGD_HPP
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
namespace nnqs
{
template<typename T>
class SGD
{
private:
	double alpha_;
	double p_;
	int t_;

public:
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	SGD(double alpha, double p = 0.5)
		: alpha_(alpha), p_(p)
	{
		t_ = 0;
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "SGD"},
			{"alhpa", alpha_}
		};
	}

	Vector getUpdate(const Vector& v)
	{
		using std::pow;
		++t_;
		double eta = std::max((alpha_/pow(t_, p_)), 1e-4);
		return -eta*v;
	}

};
}//namespace nnqs

#endif//CY_SGD_HPP
