#ifndef YANNQ_MACHINES_AMPLITUDEPHASE_HPP
#define YANNQ_MACHINES_AMPLITUDEPHASE_HPP
#include <torch/torch.h>
#include <memory>
#include "RBM.hpp"
#include "FeedForward.hpp"
namespace yannq
{

class AmplitudePhase
{
public:
	using ScalarType = double;
private:
	const uint32_t N_;

	RBM<ScalarType, false> amplitude_;
	std::shared_ptr<torch::nn::Module> phaseNet_;

	torch::TensorOptions opts_;
	torch::Device device_;

public:
	using CxScalarType = std::complex<double>;
	using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorType = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

	using CxMatrixType = Eigen::Matrix<CxScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using CxVectorType = Eigen::Matrix<CxScalarType, Eigen::Dynamic, 1>;
	using AmpDataType = std::tuple<Eigen::VectorXi, VectorType>;
	using PhaseDataType = std::vector<VectorType>;

	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

	AmplitudePhase(uint32_t N, uint32_t M, std::shared_ptr<torch::nn::Module> phaseNet)
		: N_{N}, amplitude_(N, M), phaseNet_(std::move(phaseNet))
	{
		torch::DeviceType device_type;
		if (torch::cuda::is_available()) {
			std::cout << "CUDA available! Training on GPU." << std::endl;
			device_type = torch::kCUDA;
		} else {
			std::cout << "Training on CPU." << std::endl;
			device_type = torch::kCPU;
		}
		device_  = device_type;

		phaseNet_->to(device);
		
		opt_ = torch::TensorOptions().dtype(torch::kFloat64).device(device_type);
	}

	/*!
	 * input: a matrix of configurations (N x nSmp)
	 * returns a tuple of logCoeffs (CxVector), logDeriv of amplidue, and logDeriv of phase
	 */
	std::tuple<CxVectorType, VectorType, VectorType>
	logDeriv(const MatrixType& sigmas) 
	{
		MatrixType sigmas(N_, ampData.size());
#pragma omp parallel for schedule(static, 8)
		for(uint32_t i = 0; i < ampData.size(); i++)
		{
			sigmas.col(i) = std::get<0>(ampData[i]);
		}
	}

	inline uint32_t getDimAmp() const
	{
		return amplitude_.getDim();
	}

	std::complex<ScalarType> logCoeff(const AmpDataType& data) const
	{
		CxScalarType res{};
		res.real = amp_.logCoeff(data);
		res.imag = 
	}

	ScalarType phaseForward(const Eigen::VectorXi& sigma) const
	{
		Eigen::VectorXd s = sigma.cast<ScalarType>();
		torch::Tensor s_t = torch::from_blob(s.data(), {1,1,1,N_}, opt.requires_grad(false));
		torch::Tensor phase_t = phaseNet_->forward(s_t);
		phaseRes_t = phaseRes_t.to(torch::kCPU);

		auto phaseRes_a = phasRes_t.accessor<double,1>();

		for(int i = 0; i < phaseRes_a.size(0); i++) 
		{
			res[idx].imag = M_PI*phaseRea_a[i];
		}
		return res;

	}

	CxVectorType logCoeffs(const std::vector<AmpDataType>& ampData)
	{
		auto nSmp = ampData.size();
		CxVectorType res{};
#pragma omp parallel for schedule(static, 8)
		for(uint32_t idx = 0; idx < ampData.size(); idx++)
		{
			res[idx].real = std::log(amp_.coeff(ampData[idx]))/2.0;
		}

		MatrixType sigmas(N_, ampData.size());
#pragma omp parallel for schedule(static, 8)
		for(uint32_t i = 0; i < ampData.size(); i++)
		{
			sigmas.col(i) = std::get<0>(ampData[i]);
		}

		torch::Tensor sigmas_t = torch::from_blob(sigmas.data(), {nSmp, 1, 1, N_}, opt_.requires_grad(false));
		torch::Tensor phaseRes_t = phaseNet_->forward(sigmas_t);
		phaseRes_t = phaseRes_t.to(torch::kCPU);

		auto phaseRes_a = phasRes_t.accessor<double,1>();

		for(int i = 0; i < phaseRes_a.size(0); i++) 
		{
			res[idx].imag = M_PI*phaseRea_a[i];
		}
		return res;
	}

	template<typename RandomEngine>
	void initializeAmplitudeRandom(RandomEngine&& re, double sigma)
	{
		amplitude_.initializeRandom(re, sigma);
	}

	inline AmpDataType makeAmpData(const Eigen::VectorXi& sigma) const
	{
		return amplitude_.makeData(sigma);
	}

	std::pair<AmpDataType,PhaseDataType> makeData(const Eigen::VectorXi& sigma) const
	{
		return std::make_pair(amplitude_.makeData(sigma), phase_.makeData(sigma));
	}

	CxScalarType coeff(const AmpDataType& t) const
	{
		constexpr std::complex<double> I(0.,1.);
		return std::sqrt(amplitude_.coeff(t))*std::exp(I*M_PI*phase_.forward(std::get<0>(t)));
	}

	uint32_t getN() const
	{
		return N_;
	}

	void updateParams(const VectorConstRefType& m)
	{
		assert(m.size() == getDim());
		amplitude_.updateParams(m.head(amplitude_.getDim()));
		phase_.updateParams(m.segment(amplitude_.getDim(), phase_.getDim()));
	}
};

AmplitudePhase::CxVectorType getPsi(const AmplitudePhase& qs, bool normalize)
{
	const auto n = qs.getN();
	AmplitudePhase::CxVectorType psi(1<<n);
#pragma omp parallel for schedule(static,8)
	for(uint64_t i = 0; i < (1u<<n); i++)
	{
		auto s = toSigma(n, i);
		psi(i) = qs.coeff(qs.makeAmpData(s));
	}
	if(normalize)
		return psi.normalized();
	else
		return psi;
}

AmplitudePhase::CxVectorType getPsi(const AmplitudePhase& qs, const std::vector<uint32_t>& basis, bool normalize)
{
	const auto n = qs.getN();
	AmplitudePhase::CxVectorType psi(basis.size());
#pragma omp parallel for schedule(static,8)
	for(uint64_t i = 0; i < basis.size(); i++)
	{
		auto s = toSigma(n, basis[i]);
		psi(i) = qs.coeff(qs.makeAmpData(s));
	}
	if(normalize)
		return psi.normalized();
	else
		return psi;
}
}//namespace yannq
#endif//YANNQ_MACHINES_AMPLITUDEPHASE_HPP
