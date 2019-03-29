#ifndef NNQS_OPTIMIZERS_OPTIMIZER_HPP
#define NNQS_OPTIMIZERS_OPTIMIZER_HPP
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "Utilities/type_traits.hpp"

namespace nnqs
{
template<typename T>
class Optimizer
{
public:
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using RealVector = Eigen::Matrix<typename nnqs::remove_complex<T>::type, Eigen::Dynamic, 1>;

	virtual nlohmann::json params() const = 0; 
	virtual Vector getUpdate(const Vector& ) = 0;
};
}
#endif//NNQS_OPTIMIZERS_OPTIMIZER_HPP
