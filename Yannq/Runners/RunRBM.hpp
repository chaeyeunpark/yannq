#ifndef YANNQ_RUNNERS_RUNRBM_HPP
#define YANNQ_RUNNERS_RUNRBM_HPP

#include "AbstractRunner.hpp"

#include "Samplers/Sampler.hpp"
#include "Samplers/SamplerPT.hpp"
#include "Samplers/ExactSampler.hpp"

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

//! \addtogroup Runners

//! \ingroup Runners
template<typename T, class RandomEngine = std::default_random_engine>
class RunRBM
	: public AbstractRunner<RBM<T>, RandomEngine, RunRBM<T, RandomEngine>>
{
public:
	using MachineType = RBM<T>;

private:
	bool useCG_ = false;
	double cgTol_ = 1e-4;

	double beta1_ = 0.0;
	double beta2_ = 0.0;

public:
	RunRBM(const unsigned int N, const unsigned int alpha,
			bool useBias, std::ostream& logger)
		: AbstractRunner<RBM<T>, RandomEngine, RunRBM<T, RandomEngine>>
		  	(logger, N, /* M = */alpha*N, useBias)
	{
	}

	void setSolverParams(bool useCG, double cgTol = 1e-4)
	{
		useCG_ = useCG;
		cgTol_ = cgTol;
	}

	void setMomentum(double beta1 = 0.0, double beta2 = 0.0)
	{
		beta1_ = beta1;
		beta2_ = beta2;
	}

	//if usePT
	template<class Sweeper> 
	auto createSamplerPT(Sweeper& sweeper, uint32_t numChains) const
	{
		return SamplerPT<MachineType, std::default_random_engine, RBMStateValue<T>, Sweeper>(this->qs_, numChains, sweeper);
	}
	//if not usePT
	template<class Sweeper> 
	auto createSamplerMT(Sweeper& sweeper) const
	{
		return Sampler<MachineType, std::default_random_engine, RBMStateValueMT<T>, Sweeper>(this->qs_, sweeper);
	}
	
	template<class Iterable>
	auto createSamplerExact(Iterable&& basis) const
	{
		return ExactSampler<MachineType, std::default_random_engine>(this->qs_, std::forward<Iterable>(basis));
	}
	
	/** \brief Run the calculation
	 *
	 * Two template parameters determine which Sweeper to use and whether to use the parallel tempering.
	 * For \f$U(1)\f$ symmetric Hamiltonains, you may use the SwapSweeper. Otherwise, LocalSweeper should be used.
	 * 
	 * \param sampler Any sampler can be taken. Usually, it is convinient to use createSampler methods.
	 * \param callback Callback function that is called for each epoch. The parameters of the callback function.
	 * is given by (the epoch, estimated energy in this epoch, norm of the update, error from conjugate gradient solver, sampling duration, solving duration).
	 * \param randomizer It is a functor that intiailize the sampler before each use.
	 * \param ham The Hamiltonian we want to solve.
	 * \param nSamples The number of samples we will use for each epoch. Default: the number of parameters of the machine.
	 * \param nSamplesDiscard The number of samples we will discard before sampling. It is used for equilbration. Default: 0.1*nSamples.
	 */
	template<class Sampler, class Callback, class SweeperRandomizer, class Hamiltonian>
	void run(Sampler& sampler, Callback&& callback, SweeperRandomizer&& randomizer, Hamiltonian&& ham, int nSamples = -1, int nSamplesDiscard = -1)
	{
		using Clock = std::chrono::high_resolution_clock;
		using namespace yannq;
		using Matrix = typename MachineType::Matrix;
		using Vector = typename MachineType::Vector;

		if(!this->threadsInitiialized_)
			this->initializeThreads();
		if(!this->weightsInitialized_)
			this->initializeRandom();

		SRMat<MachineType,Hamiltonian> srm(this->qs_, std::forward<Hamiltonian>(ham));
		
		sampler.initializeRandomEngine();

		const int dim = this->getDim();

		if(nSamples == -1)
			nSamples = dim;
		if(nSamplesDiscard == -1)
			nSamplesDiscard = int(0.1*nSamples);

		ObsAvg<Vector> gradAvg(beta1_, Vector::Zero(dim));
		ObsAvg<Matrix> fisherAvg(beta2_, Matrix::Zero(dim,dim));

		//These should be changed into structured binding in C++17
		double lambdaIni, lambdaDecay, lambdaMin;
		std::tie(lambdaIni, lambdaDecay, lambdaMin) 
			= this->getLambdas();
		int maxIter, saveWfPer;
		std::tie(maxIter, saveWfPer) = this->getIterParams();

		for(int ll = 0; ll <= maxIter; ll++)
		{
			this->logger() << "Epochs: " << ll << "\t# of samples:" << nSamples << 
				"\t# of discard samples: " << nSamplesDiscard << std::endl;
			if((saveWfPer != 0) && (ll % saveWfPer == 0))
			{
				char fileName[30];
				sprintf(fileName, "w%04d.dat",ll);
				std::fstream out(fileName, std::ios::binary | std::ios::out);
				{
					auto qsToSave = std::make_unique<MachineType>(this->qs_);
					cereal::BinaryOutputArchive oa(out);
					oa(qsToSave);
				}
			}
			randomizer(sampler);

			//Sampling
			auto smp_start = Clock::now();
			auto sr = sampler.sampling(nSamples, nSamplesDiscard);
			auto smp_dur = std::chrono::duration_cast<std::chrono::milliseconds>
				(Clock::now() - smp_start).count();

			auto slv_start = Clock::now();

			srm.constructFromSampling(sr);

			double currE = srm.eloc();
			
			double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
			Vector v;
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
				fisher += lambda*Matrix::Identity(dim,dim);
				Eigen::LLT<Matrix> llt{fisher};
				v = llt.solve(gradAvg.getAvg());
				cgErr = 0;
			}

			Vector optV = this->opt_->getUpdate(v);

			auto slv_dur = std::chrono::duration_cast<std::chrono::milliseconds>
				(Clock::now() - slv_start).count();

			double nv = v.norm();

			this->qs_.updateParams(optV);

			callback(ll, currE, nv, cgErr, smp_dur, slv_dur);
		}
	}
};
} //namespace yannq
#endif//YANNQ_RUNNERS_RUNRBM_HPP
