#pragma once
#include "AbstractRunner.hpp"
#include "Supervised/OverlapOptimizerExact.hpp"
#include "GroundState/SRMatExact.hpp"

namespace yannq
{
template<typename T, template<typename > class MachineT,
	class RandomEngine = std::default_random_engine>
class RunExact
	: public AbstractRunner<MachineT<T>, RandomEngine>
{
public:
	using Machine = MachineT<T>;
	using Matrix = typename Machine::Matrix;
	using Vector = typename Machine::Vector;

public:
	template<typename... Ts>
	RunExact(std::ostream& logger, Ts&&... args)
		: AbstractRunner<Machine, RandomEngine>
		  	(logger, std::forward<Ts>(args)...)
	{
	}

	nlohmann::json getAdditionalParams() const override
	{
		return R"({"name": "RunExact"})"_json;
	}

	template<class Callback, class Basis, class Hamiltonian>
	void run(Callback&& callback, Basis&& basis, Hamiltonian&& ham)
	{
		using std::pow;
		using std::max;
		using namespace yannq;
		using Clock = std::chrono::high_resolution_clock;

		if(!this->machineInitialized())
			this->initializeRandom();

		const int dim = this->getDim();

		const auto [lambdaIni, lambdaDecay, lambdaMin]
			= this->getLambdas();
		int maxIter, saveWfPer;
		std::tie(maxIter, saveWfPer) = this->getIterParams();

		SRMatExact<Machine> srex(this->qs_, std::forward<Basis>(basis), ham);

		for(int ll = 0; ll <= maxIter; ll++)
		{
			this->logger() << "Epochs: " << ll << std::endl;
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
			srex.clear();
			srex.constructExact();

			double currE = srex.eloc();
			auto corrMat = srex.corrMat();
			double lambda = std::max(lambdaIni*pow(lambdaDecay,ll), lambdaMin);
			corrMat += lambda*Matrix::Identity(dim,dim);
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
