#ifndef YANNQ_RUNNERS_RUNRBMEXACT_HPP
#define YANNQ_RUNNERS_RUNRBMEXACT_HPP

#include "AbstractRunner.hpp"
#include "Supervised/OverlapOptimizerExact.hpp"
#include "GroundState/SRMatExact.hpp"

namespace yannq
{
template<typename T, class RandomEngine = std::default_random_engine>
class RunRBMExact
	: public AbstractRunner<T, RandomEngine, RunRBMExact<T, RandomEngine> >
{
public:
	using MachineT = typename AbstractRunner<T, RandomEngine, RunRBMExact<T, RandomEngine>>::MachineT;
	using MatrixType = typename MachineT::MatrixType;
	using VectorType = typename MachineT::VectorType;

public:
	RunRBMExact(const uint32_t N, const int alpha, bool useBias, std::ostream& logger)
		: AbstractRunner<T, RandomEngine, RunRBMExact<T, RandomEngine>>
		  	(N, alpha, useBias, logger)
	{
	}

	template<class Callback, class Basis, class Hamiltonian>
	void run(Callback&& callback, Basis&& basis, Hamiltonian&& ham)
	{
		using std::pow;
		using std::max;
		using namespace yannq;
		using Clock = std::chrono::high_resolution_clock;

		if(!this->threadsInitiialized_)
			this->initializeThreads();
		if(!this->weightsInitialized_)
			this->initializeRandom();

		const int dim = this->getDim();

		//In C++17, these should be changed into structured binding
		double lambdaIni, lambdaDecay, lambdaMin;
		std::tie(lambdaIni, lambdaDecay, lambdaMin) 
			= this->getLambdas();
		int maxIter, saveWfPer;
		std::tie(maxIter, saveWfPer) = this->getIterParams();

		SRMatExact<MachineT> srex(this->qs_, std::forward<Basis>(basis), ham);

		for(int ll = 0; ll <= maxIter; ll++)
		{
			this->logger() << "Epochs: " << ll << std::endl;
			if((saveWfPer != 0) && (ll % saveWfPer == 0))
			{
				char fileName[30];
				sprintf(fileName, "w%04d.dat",ll);
				std::fstream out(fileName, std::ios::binary | std::ios::out);
				{
					auto qsToSave = std::make_unique<MachineT>(this->qs_);
					cereal::BinaryOutputArchive oa(out);
					oa(qsToSave);
				}
			}

			srex.constructExact();

			double currE = srex.eloc();
			auto corrMat = srex.corrMat();
			double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
			corrMat += lambda*MatrixType::Identity(dim,dim);
			Eigen::LLT<Eigen::MatrixXcd> llt(corrMat);

			auto grad = srex.energyGrad();
			auto v = llt.solve(grad);
			auto optV = this->opt_->getUpdate(v);

			this->qs_.updateParams(optV);
			double nv = v.norm();

			callback(ll, currE, nv);
		}
	}

	template<class Callback, class Basis>
	void runSupervised(Callback&& callback, Basis&& basis, const VectorType& st)
	{
		using std::pow;
		using std::max;
		using namespace yannq;
		using Clock = std::chrono::high_resolution_clock;

		if(!this->threadsInitiialized_)
			this->initializeThreads();
		if(!this->weightsInitialized_)
			this->initializeRandom();

		const int dim = this->getDim();

		//In C++17, these should be changed into structured binding
		double lambdaIni, lambdaDecay, lambdaMin;
		std::tie(lambdaIni, lambdaDecay, lambdaMin) 
			= this->getLambdas();
		int maxIter, saveWfPer;
		std::tie(maxIter, saveWfPer) = this->getIterParams();

		OverlapOptimizerExact<MachineT> ovex(this->qs_, std::forward<Basis>(basis));

		ovex.setTarget(st);

		for(int ll = 0; ll <= maxIter; ll++)
		{
			this->logger() << "Epochs: " << ll << std::endl;
			if((saveWfPer != 0) && (ll % saveWfPer == 0))
			{
				char fileName[30];
				sprintf(fileName, "w%04d.dat",ll);
				std::fstream out(fileName, std::ios::binary | std::ios::out);
				{
					auto qsToSave = std::make_unique<MachineT>(this->qs_);
					cereal::BinaryOutputArchive oa(out);
					oa(qsToSave);
				}
			}

			ovex.constructExact();

			double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
			auto corrMat = ovex.corrMat();
			corrMat += lambda*MatrixType::Identity(dim,dim);
			Eigen::LLT<Eigen::MatrixXcd> llt(corrMat);
			auto grad = ovex.calcLogGrad();
			auto v = llt.solve(grad);
			double nv = v.norm();

			auto optV = this->opt_->getUpdate(v);
			double fidelity = ovex.fidelity();

			this->qs_.updateParams(optV);

			callback(ll, fidelity, nv);
		}
	}
};
} //namespace yannq
#endif//YANNQ_RUNNERS_RUNRBMEXACT_HPP
