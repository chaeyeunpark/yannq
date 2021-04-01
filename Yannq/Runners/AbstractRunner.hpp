#pragma once
#include <random>
#include <chrono>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <nlohmann/json.hpp>
#include <ios>

#include "filesystem.hpp"
#include "Yannq.hpp"
#include "Serializers/SerializeRBM.hpp"
namespace yannq
{
template<class Machine, class RandomEngine>
class AbstractRunner
{
public:
	using json = nlohmann::json;

	using path = fs::path;

protected:
	std::ostream& logger_;
	RandomEngine re_;
	Machine qs_;


private:
	bool threadsInitialized_ = false;
	bool machineInitialized_ = false;

	double lambdaIni_ = 1e-3;
	double lambdaDecay_ = 1.0;
	double lambdaMin_ = 1e-4;

	uint32_t maxIter_ = 2000;
	uint32_t saveWfPer_ = 100;
	uint32_t numChains_ = 16;

protected:
	std::unique_ptr<yannq::Optimizer<typename Machine::Scalar> > opt_;

	template<typename ...Ts>
	AbstractRunner(std::ostream& logger, Ts... args)
		: logger_{logger},
		qs_(std::forward<Ts>(args)...)
	{
		std::random_device rd;
		re_.seed(rd());
	}

	virtual json getAdditionalParams() const = 0;


public:

	std::ostream& logger()
	{
		return logger_;
	}
	const std::ostream& logger() const
	{
		return logger_;
	}

	uint32_t getDim() const
	{
		return qs_.getDim();
	}

	void setLambda(double lambdaIni, double lambdaDecay = 1.0, double lambdaMin = 0.0)
	{
		lambdaIni_ = lambdaIni;
		lambdaDecay_ = lambdaDecay;
		lambdaMin_ = lambdaMin;
	}
	
	std::tuple<double, double, double> getLambdas() const
	{
		return {lambdaIni_, lambdaDecay_, lambdaMin_};
	}

	void initializeFrom(const path& filePath)
	{
		using std::ios;
		machineInitialized_ = true;
		logger_ << "Loading initial weights from " << filePath << std::endl;

		std::fstream in(filePath, ios::binary | ios::in);
		cereal::BinaryInputArchive ia(in);
		std::unique_ptr<Machine> qsLoad{nullptr};
		ia(qsLoad);
		qs_ = *qsLoad;
	}

	void initializeRandom(double wIni = 1e-3)
	{
		machineInitialized_ = true;
		logger_ << "Set initial weights randomly from N(0.0, " << wIni << "^2)" << std::endl;
		qs_.initializeRandom(re_, wIni);
	}

	bool threadsInitialized() const
	{
		return threadsInitialized_;
	}

	bool machineInitialized() const
	{
		return machineInitialized_;
	}

	void setOptimizer(const json& optParam)
	{
		opt_ = std::move(yannq::OptimizerFactory<typename Machine::Scalar>::getInstance().createOptimizer(optParam));
	}

	void setIterParams(const int maxIter, const int saveWfPer)
	{
		maxIter_ = maxIter;
		saveWfPer_ = saveWfPer;
	}

	std::pair<int, int> getIterParams() const
	{
		return {maxIter_, saveWfPer_};
	}

	const Machine& getQs() const &
	{
		return qs_;
	}
	Machine getQs() && 
	{
		return qs_;
	}

	json getParams() const
	{
		json j;
		j["Optimizer"] = opt_->desc();
		
		json SR = 
		{
			{"lambdaIni", lambdaIni_},
			{"lambdaDecay", lambdaDecay_},
			{"lambdaMin", lambdaMin_},
		};
		j["SR"] = SR;
		j["numThreads"] = Eigen::nbThreads();
		j["machine"] = qs_.desc();

		json to_update = this->getAdditionalParams();
		if(!to_update.is_null())
			j.update(to_update);

		return j;
	}

};
}// namespace yannq
