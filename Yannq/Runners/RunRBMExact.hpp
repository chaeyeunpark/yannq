#ifndef YANNQ_RUNNERS_RUNRBMEXACT_HPP
#define YANNQ_RUNNERS_RUNRBMEXACT_HPP
#include <chrono>

#include <boost/filesystem.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <nlohmann/json.hpp>
#include <ios>

#include "Yannq.hpp"

namespace yannq
{
template<typename T, bool useBias, class RandomEngine = std::default_random_engine>
class RunRBMExact
{
public:
	using MachineT = yannq::RBM<T, useBias>;
	using json = nlohmann::json;

private:
	MachineT qs_;
	std::ostream& logger_;
	RandomEngine re_;

	bool useCG_ = false;
	double lambdaIni_ = 1e-3;
	double lambdaDecay_ = 1.0;
	double lambdaMin_ = 1e-4;

	int maxIter_ = 2000;
	int saveWfPer_ = 100;
	int numChains_ = 16;

	std::unique_ptr<yannq::Optimizer<T> > opt_;

public:
	RunRBMExact(const uint32_t N, const int alpha, std::ostream& logger)
		: qs_(N, alpha*N), logger_{logger}
	{
		std::random_device rd;
		re_.seed(rd());
	}

	void setLambda(double lambdaIni, double lambdaDecay, double lambdaMin)
	{
		lambdaIni_ = lambdaIni;
		lambdaDecay_ = lambdaDecay;
		lambdaMin_ = lambdaMin;
	}

	void initializeFrom(const boost::filesystem::path& path)
	{
		using std::ios;
		logger_ << "Loading initial weights from " << path << std::endl;

		boost::filesystem::fstream in(path, ios::binary | ios::in);
		cereal::BinaryInputArchive ia(in);
		std::unique_ptr<MachineT> qsLoad{nullptr};
		ia(qsLoad);
		qs_ = *qsLoad;
	}

	void initializeRandom(double wIni)
	{
		logger_ << "Set initial weights randomly from N(0.0, " << wIni << "^2)" << std::endl;
		qs_.initializeRandom(re_, wIni);
	}

	void setOptimizer(const json& optParam)
	{
		opt_ = std::move(yannq::OptimizerFactory<T>::getInstance().createOptimizer(optParam));
	}

	void setIterParams(const int maxIter, const int saveWfPer)
	{
		maxIter_ = maxIter;
		saveWfPer_ = saveWfPer;
	}

	const MachineT& getQs() const &
	{
		return qs_;
	}
	MachineT getQs() && 
	{
		return qs_;
	}

	json getParams() const
	{
		json j;
		j["Optimizer"] = opt_->params();
		
		json SR = 
		{
			{"lambdaIni", lambdaIni_},
			{"lambdaDecay", lambdaDecay_},
			{"lambdaMin", lambdaMin_},
		};
		j["SR"] = SR;
		j["numThreads"] = Eigen::nbThreads();
		j["machine"] = qs_.params();
		j["sampler"] = 
		{
			{"PT", numChains_},
		};

		return j;
	}

	template<class Callback, class Basis, class Hamiltonian>
	void run(Callback&& callback, Basis&& basis, Hamiltonian&& ham)
	{
		using std::pow;
		using std::max;
		using namespace yannq;
		using Clock = std::chrono::high_resolution_clock;
		using MatrixT = typename MachineT::MatrixType;
		using VectorT = typename MachineT::VectorType;

		const int dim = qs_.getDim();
		SRMatExact<MachineT> srex(qs_, std::forward<Basis>(basis), ham);

		for(int ll = 0; ll <= maxIter_; ll++)
		{
			logger_ << "Epochs: " << ll << std::endl;
			if((saveWfPer_ != 0) && (ll % saveWfPer_ == 0))
			{
				char fileName[30];
				sprintf(fileName, "w%04d.dat",ll);
				std::fstream out(fileName, std::ios::binary | std::ios::out);
				{
					auto qsToSave = std::make_unique<MachineT>(qs_);
					cereal::BinaryOutputArchive oa(out);
					oa(qsToSave);
				}
			}

			srex.constructExact();

			double currE = srex.eloc();
			auto corrMat = srex.corrMat();
			double lambda = std::max(lambdaIni_*pow(lambdaDecay_,ll), lambdaMin_);
			corrMat += lambda*MatrixT::Identity(dim,dim);
			Eigen::LLT<Eigen::MatrixXcd> llt(corrMat);

			auto grad = srex.energyGrad();
			auto v = llt.solve(grad);
			auto optV = opt_->getUpdate(v);

			qs_.updateParams(optV);

			//double cgErr = (corrMat*v-srex.energyGrad()).norm();
			double nv = v.norm();

			qs_.updateParams(optV);

			callback(ll, currE, nv);
		}
	}
};
} //namespace yannq
#endif//YANNQ_RUNNERS_RUNRBMEXACT_HPP
