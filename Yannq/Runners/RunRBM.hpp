#ifndef YANNQ_RUNNERS_RUNRBM_HPP
#define YANNQ_RUNNERS_RUNRBM_HPP

#include "AbstractRunner.hpp"

#include "Samplers/Sampler.hpp"
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
	: public AbstractRunner<RBM<T>, RandomEngine>
{
public:
	using Machine = RBM<T>;

private:
	bool useCG_ = false;
	double cgTol_ = 1e-4;

	double beta1_ = 0.0;
	double beta2_ = 0.0;

	double gradClip_ = 0.0;

public:
	RunRBM(const unsigned int N, const unsigned int alpha,
			bool useBias, std::ostream& logger)
		: AbstractRunner<Machine, RandomEngine>(logger, N, alpha*N, useBias)
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

	void setGradClip(double gradClip)
	{
		gradClip_ = gradClip;
	}

	nlohmann::json getAdditionalParams() const override
	{
		using json = nlohmann::json;
		json j;
		j["use_cg"] = useCG_;
		if(useCG_)
			j["cg_tol"] = cgTol_;

		j["momentum"] = json{
			{"beta1", beta1_},
			{"beta2", beta2_},
		};
		return j;
	}

	template<class Sweeper> 
	auto createSampler(Sweeper& sweeper, uint32_t nTmps, uint32_t nChainsPer) const
	{
		return SamplerMT<Machine, RBMStateValue<T>, Sweeper>(this->qs_, nTmps, nChainsPer, sweeper);
	}
	
	template<class Iterable>
	auto createExactSampler(Iterable&& basis) const
	{
		return ExactSampler<Machine, std::default_random_engine>(this->qs_, std::forward<Iterable>(basis));
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
	 * \param nSweeps The number of samples we will use for each epoch. Default: the number of parameters of the machine.
	 * \param nSweepsDiscard The number of samples we will discard before sampling. It is used for equilbration. Default: 0.1*nSweeps.
	 */
	template<class Sampler, class Callback, class SweeperRandomizer, class Hamiltonian>
	void run(Sampler& sampler, Callback&& callback, SweeperRandomizer&& randomizer, Hamiltonian&& ham, int nSweeps = -1, int nSweepsDiscard = -1)
	{
		using Clock = std::chrono::high_resolution_clock;
		using namespace yannq;
		using Matrix = typename Machine::Matrix;
		using Vector = typename Machine::Vector;

		this->initializeRunner();

		SRMat<Machine,Hamiltonian> srm(this->qs_, std::forward<Hamiltonian>(ham));
		sampler.initializeRandomEngine();

		const int dim = this->getDim();

		if(nSweeps == -1)
			nSweeps = dim;
		if(nSweepsDiscard == -1)
			nSweepsDiscard = int(0.1*nSweeps);

		ObsAvg<Vector> gradAvg(beta1_, Vector::Zero(dim));
		ObsAvg<Matrix> fisherAvg(beta2_, Matrix::Zero(dim,dim));

		auto [lambdaIni, lambdaDecay, lambdaMin]
			= this->getLambdas();
		int maxIter, saveWfPer;
		std::tie(maxIter, saveWfPer) = this->getIterParams();

		for(int ll = 0; ll <= maxIter; ll++)
		{
			this->logger() << "Epochs: " << ll << 
				"\t# of discard samples: " << nSweepsDiscard << std::endl;
			if((saveWfPer != 0) && (ll % saveWfPer == 0))
			{
				char fileName[30];
				sprintf(fileName, "w%04d.dat",ll);
				std::fstream out(fileName, std::ios::binary | std::ios::out);
				{
					auto qsToSave = std::make_unique<Machine>(this->qs_);
					cereal::BinaryOutputArchive oa(out);
					oa(qsToSave);
				}
			}
			randomizer(sampler);

			//Sampling
			auto smp_start = Clock::now();
			auto sr = sampler.sample(nSweeps, nSweepsDiscard);
			auto smp_dur = std::chrono::duration_cast<std::chrono::milliseconds>
				(Clock::now() - smp_start).count();

			this->logger() << "Number of samples: " << sr.size() << std::endl;

			auto slv_start = Clock::now();

			srm.constructFromSamples(sr);

			double currE = srm.eloc();
			
			double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
			Vector v;
			double cgErr;

			gradAvg.update(srm.energyGrad());
			Vector grad = gradAvg.getAvg();

			if(useCG_ && (beta2_ == 0.0))
			{
				v = srm.solveCG(grad, lambda, cgTol_);
				cgErr = (srm.apply(v)-grad).norm();
			}
			else
			{
				gradAvg.update(srm.energyGrad());
				fisherAvg.update(srm.corrMat());
				auto fisher = fisherAvg.getAvg();
				fisher += lambda*Matrix::Identity(dim,dim);
				Eigen::LLT<Matrix> llt{fisher};
				v = llt.solve(grad);
				cgErr = 0;
			}

			double nv = v.norm();
			if((gradClip_ != 0.0) && (nv > gradClip_))
			{
				v *= gradClip_/nv;
			}

			Vector optV = this->opt_->getUpdate(v);

			auto slv_dur = std::chrono::duration_cast<std::chrono::milliseconds>
				(Clock::now() - slv_start).count();


			this->qs_.updateParams(optV);

			callback(ll, currE, nv, cgErr, smp_dur, slv_dur);
		}
	}
};
} //namespace yannq
#endif//YANNQ_RUNNERS_RUNRBM_HPP
