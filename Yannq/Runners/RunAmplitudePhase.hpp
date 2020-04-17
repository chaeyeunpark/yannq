#pragma once

#include "AbstractRunner.hpp"
#include "GroundState/NGDExact.hpp"

namespace yannq
{
template<class RandomEngine = std::default_random_engine>
class RunAmplitudePhaseExact
	: public AbstractRunner<AmplitudePhase, RandomEngine, RunAmplitudePhaseExact<RandomEngine> >
{
public:
	using ScalarType = typename AmplitudePhase::ScalarType;
	using MachineType = AmplitudePhase;

	using ReScalarType = typename MachineType::ReScalarType;
	using CxScalarType = typename MachineType::CxScalarType;

	using ReMatrixType = typename MachineType::ReMatrixType;
	using ReVectorType = typename MachineType::ReVectorType;

	using CxMatrixType = typename MachineType::CxMatrixType;
	using CxVectorType = typename MachineType::CxVectorType;

public:
	RunAmplitudePhaseExact(const uint32_t N, const int alpha, bool useBias, 
			FeedForward<ScalarType>&& phase,
			std::ostream& logger)
		: AbstractRunner<AmplitudePhase, RandomEngine, RunAmplitudePhaseExact<RandomEngine>>
		  	(logger, N, N*alpha, useBias, std::move(phase))
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
		const int dimAmp = (this->qs_).getDimAmp();
		const int dimPhase = (this->qs_).getDimPhase();

		//In C++17, these should be changed into structured binding
		double lambdaIni, lambdaDecay, lambdaMin;
		std::tie(lambdaIni, lambdaDecay, lambdaMin) 
			= this->getLambdas();
		int maxIter, saveWfPer;
		std::tie(maxIter, saveWfPer) = this->getIterParams();

		NGDExact ngdex(this->qs_, std::forward<Basis>(basis), ham);

		for(int ll = 0; ll <= maxIter; ll++)
		{
			this->logger() << "Epochs: " << ll << std::endl;
			/*
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
			*/

			ngdex.constructExact();

			ReScalarType currE = ngdex.eloc();
			auto grad = ngdex.energyGrad();
			ReVectorType v(this->qs_.getDim());

			{
				auto corrMatAmp = ngdex.corrMatAmp();
				double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
				corrMatAmp += lambda*ReMatrixType::Identity(dimAmp, dimAmp);
				Eigen::LLT<Eigen::MatrixXd> llt(corrMatAmp);
				v.head(dimAmp) = llt.solve(grad.head(dimAmp));
			}
			{
				auto corrMatPhase = ngdex.corrMatPhase();
				double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
				corrMatPhase += lambda*ReMatrixType::Identity(dimPhase, dimPhase);
				Eigen::LLT<Eigen::MatrixXd> llt(corrMatPhase);
				v.tail(dimPhase) = llt.solve(grad.tail(dimPhase));
			}

			auto optV = this->opt_->getUpdate(v);

			this->qs_.updateParams(optV);

			callback(ll, currE, grad.norm(), v.norm());
		}
	}
};
} //namespace yannq

