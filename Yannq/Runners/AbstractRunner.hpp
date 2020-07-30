#ifndef YANNQ_RUNNERS_ABSTRACTRUNNER_HPP
#define YANNQ_RUNNERS_ABSTRACTRUNNER_HPP
#include <random>
#include <chrono>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <nlohmann/json.hpp>
#include <ios>

#if __cpluscplus < 201703L
#include <experimental/filesystem>
#else
#include <filesystem>
#endif

#include "Yannq.hpp"
#include "Serializers/SerializeRBM.hpp"
namespace yannq
{
template<class Machine, class RandomEngine, class Derived>
class AbstractRunner
{
public:
	using json = nlohmann::json;

#if __cpluscplus < 201703L
	using path = std::experimental::filesystem::path;
#else
	using path = std::filesystem::path;
#endif

protected:
	Machine qs_;
	std::ostream& logger_;
	RandomEngine re_;

	bool threadsInitiialized_ = false;
	bool weightsInitialized_ = false;

private:
	double lambdaIni_ = 1e-3;
	double lambdaDecay_ = 1.0;
	double lambdaMin_ = 1e-4;

	int maxIter_ = 2000;
	int saveWfPer_ = 100;
	int numChains_ = 16;

	tbb::task_scheduler_init init_;

protected:
	std::unique_ptr<yannq::Optimizer<typename Machine::Scalar> > opt_;

	template<typename ...Ts>
	AbstractRunner(std::ostream& logger, Ts&&... args)
		: qs_(std::forward<Ts>(args)...), logger_{logger},
		init_(tbb::task_scheduler_init::deferred)
	{
		std::random_device rd;
		re_.seed(rd());
	}


public:

	std::ostream& logger()
	{
		return logger_;
	}
	const std::ostream& logger() const
	{
		return logger_;
	}

	uint32_t getDim()
	{
		return qs_.getDim();
	}

	void initializeThreads(int numThreads = tbb::task_scheduler_init::automatic)
	{
		char* var = std::getenv("TBB_NUM_THREADS");
		int val;
		if((var != nullptr) && (var[0] != '\0') && 
				(sscanf(var, "%d", &val) == 1) && (val > 0) &&
				(numThreads == tbb::task_scheduler_init::automatic))
		{
			logger_ << "Using TBB_NUM_THREADS" << std::endl;
			numThreads = val;
		}
		if(numThreads > 0)
			logger_ << "Initialize with " << numThreads << " threads" << std::endl;
		else
			logger_ << "Automatic initialize tbb threads" << std::endl;
		threadsInitiialized_ = true;
		init_.initialize(numThreads);
	}

	void setLambda(double lambdaIni, double lambdaDecay, double lambdaMin)
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
		weightsInitialized_ = true;
		logger_ << "Loading initial weights from " << filePath << std::endl;

		std::fstream in(filePath, ios::binary | ios::in);
		cereal::BinaryInputArchive ia(in);
		std::unique_ptr<Machine> qsLoad{nullptr};
		ia(qsLoad);
		qs_ = *qsLoad;
	}

	void initializeRandom(double wIni = 1e-3)
	{
		weightsInitialized_ = true;
		logger_ << "Set initial weights randomly from N(0.0, " << wIni << "^2)" << std::endl;
		qs_.initializeRandom(re_, wIni);
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

		j.update(static_cast<const Derived&>(*this).getAdditionalParams());
		return j;
	}

	json getAdditionalParams() const
	{
		static_assert("This function should be implemented in a derived class using CRTP.");
	}

	template<typename ...Ts>
	void run(Ts&&... params)
	{
		Derived::run(std::forward<Ts>(params)...);
	}
};
}// namespace yannq
#endif//YANNQ_RUNNERS_ABSTRACTRUNNER_HPP
