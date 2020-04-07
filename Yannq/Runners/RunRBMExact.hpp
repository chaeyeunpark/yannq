#ifndef YANNQ_RUNNERS_RUNRBMEXACT_HPP
#define YANNQ_RUNNERS_RUNRBMEXACT_HPP

#include "AbstractRunner.hpp"

namespace yannq
{
template<typename T, class RandomEngine = std::default_random_engine>
class RunRBMExact
	: public AbstractRunner<T, RandomEngine, RunRBMExact<T, RandomEngine> >
{
public:
	using MachineT = typename AbstractRunner<T, RandomEngine, RunRBMExact<T, RandomEngine>>::MachineT;

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
		using MatrixT = typename MachineT::MatrixType;
		using VectorT = typename MachineT::VectorType;

		if(!this->threadsInitiialized_)
			this->initializeThreads();
		if(!this->weightsInitialized_)
			this->initializeRandom();

		const int dim = this->getDim();

		//These should be changed into structured binding for C++17
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
			corrMat += lambda*MatrixT::Identity(dim,dim);
			Eigen::LLT<Eigen::MatrixXcd> llt(corrMat);

			auto grad = srex.energyGrad();
			auto v = llt.solve(grad);
			auto optV = this->opt_->getUpdate(v);

			this->qs_.updateParams(optV);
			double nv = v.norm();

			callback(ll, currE, nv);
		}
	}
};
} //namespace yannq
#endif//YANNQ_RUNNERS_RUNRBMEXACT_HPP
