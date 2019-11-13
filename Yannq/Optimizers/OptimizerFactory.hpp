#ifndef NNQS_OPTIMIZERS_OPTIMIZERFACTORY_HPP
#define NNQS_OPTIMIZERS_OPTIMIZERFACTORY_HPP
#include <functional>
#include <utility>

#include "Optimizers/Optimizer.hpp"

#include "Optimizers/SGD.hpp"
#include "Optimizers/Adam.hpp"
#include "Optimizers/AdaMax.hpp"
#include "Optimizers/AdaDelta.hpp"
#include "Optimizers/ModifiedAdam.hpp"
#include "Optimizers/RMSProp.hpp"

#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

namespace yannq
{

template<typename T>
class OptimizerFactory
{
private:
	std::unordered_map<std::string, std::function<std::unique_ptr<Optimizer<T>>(const nlohmann::json&)> > optCstr_;
	std::unordered_map<std::string, std::function<std::unique_ptr<OptimizerGeometry<T>>(const nlohmann::json&)> > optGeoCstr_;

	template<class OptimizerT>
	void resiterOptimizer(const std::string& name)
	{
		optCstr_[name] = [](const nlohmann::json& param) -> std::unique_ptr<Optimizer<T> >
		{
			return std::make_unique<OptimizerT>(param); 
		};
	}

	template<class OptimizerT>
	void resiterOptimizerGeometry(const std::string& name)
	{
		optGeoCstr_[name] = [](const nlohmann::json& param) -> std::unique_ptr<OptimizerGeometry<T> >
		{
			return std::make_unique<OptimizerT>(param); 
		};
	}

	explicit OptimizerFactory()
	{
		resiterOptimizer<SGD<T> >("SGD");

		resiterOptimizer<AdaDelta<T> >("AdaDelta");
		resiterOptimizer<RMSProp<T> >("RMSProp");
		resiterOptimizer<Adam<T> >("Adam");
		resiterOptimizer<AdaMax<T> >("AdaMax");
		resiterOptimizer<ModifiedAdam<T> >("ModifiedAdam");

		resiterOptimizer<AdaDelta<typename remove_complex<T>::type> >("AdaDeltaReal");
		resiterOptimizer<RMSProp<typename remove_complex<T>::type> >("RMSPropReal");
		resiterOptimizer<Adam<typename remove_complex<T>::type> >("AdamReal");
		resiterOptimizer<AdaMax<typename remove_complex<T>::type> >("AdaMaxReal");


		resiterOptimizerGeometry<AdaDelta<T> >("AdaDelta");
		resiterOptimizerGeometry<RMSProp<T> >("RMSProp");
		resiterOptimizerGeometry<Adam<T> >("Adam");
		resiterOptimizerGeometry<AdaMax<T> >("AdaMax");
		resiterOptimizerGeometry<ModifiedAdam<T> >("ModifiedAdam");

		resiterOptimizerGeometry<AdaDelta<typename remove_complex<T>::type> >("AdaDeltaReal");
		resiterOptimizerGeometry<RMSProp<typename remove_complex<T>::type> >("RMSPropReal");
		resiterOptimizerGeometry<Adam<typename remove_complex<T>::type> >("AdamReal");
		resiterOptimizerGeometry<AdaMax<typename remove_complex<T>::type> >("AdaMaxReal");
	}

public:
	OptimizerFactory(const OptimizerFactory& ) = delete;
	OptimizerFactory& operator=(const OptimizerFactory&) = delete;

	static OptimizerFactory& getInstance()
	{
		static OptimizerFactory instance;
		return instance;
	}

	std::unique_ptr<Optimizer<T> > createOptimizer(const nlohmann::json& opt) const
	{
		auto iter = optCstr_.find(opt["name"]);
		if (iter == optCstr_.end())
			throw std::invalid_argument("Such an optimizer does not exist.");
		return iter->second(opt);
	}

	std::unique_ptr<OptimizerGeometry<T> > createOptimizerGeometry(const nlohmann::json& opt)
	{
		auto iter = optGeoCstr_.find(opt["name"]);
		if (iter == optGeoCstr_.end())
			throw std::invalid_argument("Such an optimizer does not exist.");
		return iter->second(opt);
	}
};
}
#endif//NNQS_OPTIMIZERS_OPTIMIZERFACTORY_HPP
