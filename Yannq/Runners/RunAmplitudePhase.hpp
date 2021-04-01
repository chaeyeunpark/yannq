#pragma once
#include <cereal/archives/binary.hpp>
#include "AbstractRunner.hpp"
#include "Machines/AmplitudePhase.hpp"
#include "GroundState/NGDExact.hpp"

namespace yannq
{

enum class Mode {AMP, PHASE, ALL};

template<class RandomEngine = std::default_random_engine>
class RunAmplitudePhaseExact
	: public AbstractRunner<AmplitudePhase, RandomEngine >
{
public:
	using Scalar = typename AmplitudePhase::Scalar;
	using MachineType = AmplitudePhase;

	using RealScalar = Scalar;
	using ComplexScalar = typename MachineType::ComplexScalar;

	using RealMatrix = typename MachineType::RealMatrix;
	using RealVector = typename MachineType::RealVector;

	using ComplexMatrix = typename MachineType::ComplexMatrix;
	using ComplexVector = typename MachineType::ComplexVector;

public:
	RunAmplitudePhaseExact(const uint32_t N, const uint32_t alpha, bool useBias, 
			FeedForward<Scalar>&& phase,
			std::ostream& logger)
		: AbstractRunner<AmplitudePhase, RandomEngine>
		  	(logger, N, N*alpha, useBias, std::move(phase))
	{
	}
	
	void initializePhase()
	{
		(this->qs_).initializePhase(this->re_, InitializationMode::Xavier);
	}

	void initializeAmplitude(double sigma)
	{
		(this->qs_).initializeAmplitudeRandom(this->re_, sigma);
	}

	nlohmann::json getAdditionalParams() const override
	{
		using json = nlohmann::json;
		json j;
		return j;
	}



	template<class Callback, class Basis, class Hamiltonian>
	void run(Callback&& callback, Basis&& basis, Hamiltonian&& ham, Mode mode = Mode::ALL)
	{
		using std::pow;
		using std::max;
		using namespace yannq;
		using Clock = std::chrono::high_resolution_clock;

		this->initializeRunner();

		const int dim = this->getDim();
		const int dimAmp = (this->qs_).getDimAmp();
		const int dimPhase = (this->qs_).getDimPhase();

		//In C++17, these should be changed into structured binding
		auto [lambdaIni, lambdaDecay, lambdaMin] = this->getLambdas();
		int maxIter, saveWfPer;
		std::tie(maxIter, saveWfPer) = this->getIterParams();

		NGDExact ngdex(this->qs_, std::forward<Basis>(basis), std::forward<Hamiltonian>(ham));

		for(int ll = 0; ll <= maxIter; ll++)
		{
			this->logger() << "Epochs: " << ll << std::endl;
			if((saveWfPer != 0) && (ll % saveWfPer == 0))
			{
				char fileName[30];
				sprintf(fileName, "w%04d.dat",ll);
				std::fstream out(fileName, std::ios::binary | std::ios::out);
				{
					cereal::BinaryOutputArchive oa(out);
					oa(this->qs_);
				}
			}

			ngdex.constructExact();

			RealScalar currE = ngdex.eloc();
			auto grad = ngdex.energyGrad();
			RealVector v = RealVector::Zero(this->qs_.getDim());

			if ((mode == Mode::AMP) || (mode == Mode::ALL)) 
			{
				auto corrMatAmp = ngdex.corrMatAmp();
				double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
				corrMatAmp += lambda*RealMatrix::Identity(dimAmp, dimAmp);
				Eigen::LLT<Eigen::MatrixXd> llt(corrMatAmp);
				v.head(dimAmp) = llt.solve(grad.head(dimAmp));
			}
			if ((mode == Mode::PHASE) || (mode == Mode::ALL))
			{
				auto corrMatPhase = ngdex.corrMatPhase();
				double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
				corrMatPhase += lambda*RealMatrix::Identity(dimPhase, dimPhase);
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

