#ifndef NNQS_OPTIMIZERS_OPTIMIZER_HPP
#define NNQS_OPTIMIZERS_OPTIMIZER_HPP
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "Utilities/type_traits.hpp"

namespace yannq
{
template<typename T>
class Optimizer
{
public:
	using RealT = typename yannq::remove_complex<T>::type;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using RealVector = Eigen::Matrix<RealT, Eigen::Dynamic, 1>;

	virtual nlohmann::json params() const = 0; 

	virtual Vector getUpdate(const Vector& grad) = 0;

	virtual ~Optimizer() { }
};

template<typename T, typename Enable = void>
class OptimizerGeometry;

/* If T is complex */
template<typename T>
class OptimizerGeometry<T, typename std::enable_if<is_complex_type<T>::value>::type>
	: public Optimizer<T>
{
public:
	using typename Optimizer<T>::RealT;
	using Vector = typename Optimizer<T>::Vector;
	using RealVector = typename Optimizer<T>::RealVector;
	
	virtual Vector getUpdate(const Vector& grad, const Vector& oloc) = 0;

	virtual Vector getUpdate(const Vector& grad) override
	{
		return this->getUpdate(grad, grad);
	}
};


/* if T is real  */
template<typename T>
class OptimizerGeometry<T, typename std::enable_if<!is_complex_type<T>::value>::type>
	: public Optimizer<T>, public OptimizerGeometry<std::complex<T> >
{
public:
	using RealT = T;
	using ComplexT = std::complex<T>;
	using Vector = typename Optimizer<T>::Vector;
	using RealVector = typename Optimizer<T>::RealVector;
	using ComplexVector = typename Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>;
	
	virtual Vector getUpdate(const Vector& grad, const Vector& oloc) = 0;

	Vector getUpdate(const Vector& grad) override
	{
		return this->getUpdate(grad, grad);
	}

	ComplexVector getUpdate(const ComplexVector& grad) override
	{
		RealVector resReal = this->getUpdate((const RealVector&)Eigen::Map<RealVector>((RealT*)grad.data(), 2*grad.rows(), 1));
		return Eigen::Map<ComplexVector>((ComplexT*)resReal.data(), grad.rows(), 1);
	}

	ComplexVector getUpdate(const ComplexVector& grad, const ComplexVector& oloc) override
	{
		const auto realGrad = Eigen::Map<RealVector>((RealT*)grad.data(), 2*grad.rows(), 1);
		const auto realOloc= Eigen::Map<RealVector>((RealT*)oloc.data(), 2*oloc.rows(), 1);
		RealVector resReal = this->getUpdate((const RealVector&)realGrad, (const RealVector&)realOloc);
		return Eigen::Map<ComplexVector>((ComplexT*)resReal.data(), grad.rows(), 1);
	}
};
}
#endif//NNQS_OPTIMIZERS_OPTIMIZER_HPP
