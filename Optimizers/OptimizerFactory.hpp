#ifndef NNQS_OPTIMIZERS_OPTIMIZERFACTORY_HPP
#define NNQS_OPTIMIZERS_OPTIMIZERFACTORY_HPP
#include "Optimizers/Optimizer.hpp"

#include "Optimizers/Adam.hpp"
#include "Optimizers/AdaMax.hpp"
#include "Optimizers/SGD.hpp"

#include "Utilities/Utility.hpp"

namespace nnqs
{
template<typename T>
class OptimizerFactory
{
public:
	static std::unique_ptr<Optimizer<T> > CreateOptimizer(const nlohmann::json& opt) 
	{
		if(opt["name"].get<std::string>() == std::string("Adam"))
		{
			return make_unique<Adam<T> >(opt);
		}
		else if(opt["name"].get<std::string>() == std::string("AdaMax"))
		{
			return make_unique<AdaMax<T> >(opt);
		}
		else if(opt["name"].get<std::string>() == std::string("SGD"))
		{
			return make_unique<SGD<T> >(opt);
		}
	}
};
}
#endif//NNQS_OPTIMIZERS_OPTIMIZERFACTORY_HPP
