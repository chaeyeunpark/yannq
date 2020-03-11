#ifndef YANNQ_RUNNERS_RUNRBM_HPP
#define YANNQ_RUNNERS_RUNRBM_HPP
#include <chrono>

#include <boost/filesystem.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <nlohmann/json.hpp>
#include <ios>

#include "Yannq.hpp"

namespace yannq
{

template<class Obs>
class ObsAvg
{
private:
	int t_ = 0;
	const double beta_;
	Obs obs_{};

public:
	template<class Obs1>
	ObsAvg(double beta, Obs1&& obsIni)
		: beta_{beta}, obs_(std::forward<Obs1>(obsIni))
	{
	}
	void update(const Obs& obsNew)
	{
		obs_ = beta_*obs_ + (1.-beta_)*obsNew;
		++t_;
	}

	Obs getAvg() const
	{
		return obs_/(1.-pow(beta_,t_));
	}
};

template<typename T, bool useBias, class RandomEngine = std::default_random_engine>
class RunRBM
{
public:
	using MachineT = yannq::RBM<T, useBias>;
	using json = nlohmann::json;

private:
	MachineT qs_;
	std::ostream& logger_;
	RandomEngine re_;

	bool useCG_ = false;
	double cgTol_ = 1e-4;

	double lambdaIni_ = 1e-3;
	double lambdaDecay_ = 1.0;
	double lambdaMin_ = 1e-4;

	double beta1_ = 0.0;
	double beta2_ = 0.0;

	
	int maxIter_ = 2000;
	int saveWfPer_ = 100;
	int numChains_ = 16;

	std::unique_ptr<yannq::Optimizer<T> > opt_;

public:
	RunRBM(const uint32_t N, const int alpha, std::ostream& logger)
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
		opt_ = yannq::OptimizerFactory<T>::getInstance().createOptimizer(optParam);
	}

	void setIterParams(const int maxIter, const int saveWfPer)
	{
		maxIter_ = maxIter;
		saveWfPer_ = saveWfPer;
	}

	void setSolverParams(const bool useCG, const double cgTol = 1e-4, const double beta1 = 0.0, const double beta2 = 0.0)
	{
		useCG_ = useCG;
		cgTol_ = cgTol;
		beta1_ = beta1;
		beta2_ = beta2;
	}

	void setNumChains(const int numChains)
	{
		numChains_ = numChains;
	}

	const MachineT& getQs() const &
	{
		return qs_;
	}

	MachineT getQs() && 
	{
		return qs_;
	}

	unsigned int getDim() const
	{
		return qs_.getDim();
	}

	json getParams() const
	{
		json j;
		j["Optimizer"] = opt_->params();
		
		json SR = 
		{
			{"lambda_ini", lambdaIni_},
			{"lambda_decay", lambdaDecay_},
			{"lambda_min", lambdaMin_},
		};

		j["solver"] = {
			{"SR", SR},
			{"beta1", beta1_},
			{"beta2", beta2_}
		};
		j["numThreads"] = Eigen::nbThreads();
		j["machine"] = qs_.params();
		j["sampler"] = 
		{
			{"PT", numChains_},
		};

		return j;
	}

	//if usePT
	template<class Sweeper, bool usePT, std::enable_if_t<usePT, int> = 0 > 
	auto createSampler(Sweeper& sweeper)
	{
		return SamplerPT<MachineT, std::default_random_engine, RBMStateValue<MachineT>, Sweeper>(qs_, numChains_, sweeper);
	}
	//if not usePT
	template<class Sweeper, bool usePT, std::enable_if_t<!usePT, int> = 0 > 
	auto createSampler(Sweeper& sweeper)
	{
		return Sampler<MachineT, std::default_random_engine, RBMStateValueMT<MachineT>, Sweeper>(qs_, sweeper);
	}

	template<class Sweeper, bool usePT, class Callback, class SweeperRandomizer, class Hamiltonian>
	void run(Callback&& callback, SweeperRandomizer&& randomizer, Hamiltonian&& ham, int nSamples = -1, int nSamplesDiscard = -1)
	{
		using Clock = std::chrono::high_resolution_clock;
		using namespace yannq;
		using MatrixT = typename MachineT::MatrixType;
		using VectorT = typename MachineT::VectorType;

		Sweeper sweeper(qs_.getN());
		auto sampler = createSampler<Sweeper, usePT>(sweeper);
		SRMat<MachineT,Hamiltonian> srm(qs_, std::forward<Hamiltonian>(ham));
		
		sampler.initializeRandomEngine();

		const int dim = qs_.getDim();

		if(nSamples == -1)
			nSamples = dim;
		if(nSamplesDiscard == -1)
			nSamplesDiscard = int(0.1*nSamples);

		ObsAvg<VectorT> gradAvg(beta1_, VectorT::Zero(dim));
		ObsAvg<MatrixT> fisherAvg(beta2_, MatrixT::Zero(dim,dim));

		for(int ll = 0; ll <= maxIter_; ll++)
		{
			logger_ << "Epochs: " << ll << "\t# of samples:" << nSamples << 
				"\t# of discard samples: " << nSamplesDiscard << std::endl;
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
			randomizer(sampler);

			//Sampling
			auto smp_start = Clock::now();
			auto sr = sampler.sampling(nSamples, nSamplesDiscard);
			auto smp_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - smp_start).count();

			auto slv_start = Clock::now();

			srm.constructFromSampling(sr);

			double currE = srm.eloc();
			
			double lambda = std::max(lambdaIni_*pow(lambdaDecay_,ll), lambdaMin_);
			VectorT v;
			double cgErr;

			if(useCG_)
			{
				v = srm.solveCG(lambda, cgTol_);
				cgErr = (srm.apply(v)-srm.energyGrad()).norm();
			}
			else
			{
				gradAvg.update(srm.energyGrad());
				fisherAvg.update(srm.corrMat());
				auto fisher = fisherAvg.getAvg();
				fisher += lambda*MatrixT::Identity(dim,dim);
				Eigen::LLT<MatrixT> llt{fisher};
				v = llt.solve(gradAvg.getAvg());
				cgErr = 0;
			}
			VectorT optV = opt_->getUpdate(v);

			auto slv_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - slv_start).count();

			double nv = v.norm();

			qs_.updateParams(optV);

			callback(ll, currE, nv, cgErr, smp_dur, slv_dur);
		}
	}
};
} //namespace yannq
#endif//YANNQ_RUNNERS_RUNRBM_HPP
