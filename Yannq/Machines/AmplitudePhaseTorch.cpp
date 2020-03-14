#ifndef YANNQ_MACHINES_AMPLITUDEPHASE_HPP
#define YANNQ_MACHINES_AMPLITUDEPHASE_HPP
#include <torch/torch.h>
#include <memory>
#include <Machines/RBM.hpp>
#include <Utilities/Utility.hpp>
//#include "FeedForward.hpp"
namespace yannq
{

template<class PhaseNet>
class AmplitudePhase
{
public:
	using ScalarType = double;
private:
	const uint32_t N_;

	RBM<ScalarType, true> amplitude_;
	std::shared_ptr<PhaseNet> phaseNet_;

	torch::TensorOptions opts_;

public:
	using CxScalarType = std::complex<double>;
	using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorType = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
	using RowVectorType = Eigen::Matrix<ScalarType, 1, Eigen::Dynamic>;

	using CxMatrixType = Eigen::Matrix<CxScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using CxVectorType = Eigen::Matrix<CxScalarType, Eigen::Dynamic, 1>;
	using AmpDataType = std::tuple<Eigen::VectorXi, VectorType>;
	using PhaseDataType = std::vector<VectorType>;

	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

	AmplitudePhase(uint32_t N, uint32_t M, std::shared_ptr<PhaseNet> phaseNet)
		: N_{N}, amplitude_(N, M)
	{
		phaseNet_ = phaseNet;
		torch::DeviceType device_type;
		if (torch::cuda::is_available()) {
			std::cerr << "CUDA available! Training on GPU." << std::endl;
			device_type = torch::kCUDA;
		} else {
			std::cerr << "Training on CPU." << std::endl;
			device_type = torch::kCPU;
		}

		phaseNet_->to(device_type, torch::kFloat64);
		
		opts_ = torch::TensorOptions().dtype(torch::kFloat64).device(device_type);

	}

	/*!
	 * input: a matrix of configurations (N x nSmp)
	 * returns a tuple of logCoeffs (CxVector), logDeriv of amplidue, and logDeriv of phase
	 */
	std::tuple<CxVectorType, MatrixType, MatrixType>
	logDeriv(const Eigen::MatrixXi& sigmas) 
	{
		const unsigned int nSmp = sigmas.cols();
		MatrixType sigmas_f = sigmas.template cast<ScalarType>();
		torch::Tensor sigmas_t = torch::from_blob(sigmas_f.data(), {sigmas.cols(), 1, sigmas.rows()}, opts_);
		auto parameters = phaseNet_->parameters();
		
		unsigned int numParams = 0;
		for(const auto p: parameters)
		{
			numParams += p.numel();
		}
		torch::Tensor phaseRes_t = phaseNet_->forward(sigmas_t);
		
		torch::Tensor phaseRes_ct = phaseRes_t.to(torch::kCPU);
		auto phaseRes_a = phaseRes_ct.accessor<double,2>();

		std::vector<AmpDataType> ampDatas;
		ampDatas.reserve(nSmp);
		for(unsigned int n = 0; n < nSmp; ++n)
		{
			ampDatas.emplace_back(amplitude_.makeData(sigmas.col(n)));
		}

		CxVectorType coeffs(nSmp);
		for(unsigned int n = 0; n < nSmp; ++n)
		{
			coeffs(n) = {amplitude_.logCoeff(ampDatas[n])/2.0, M_PI*phaseRes_a[n][0]};
		}

		MatrixType ampLogDers(nSmp, amplitude_.getDim());
		for(unsigned int n = 0; n < nSmp; ++n)
		{
			ampLogDers.row(n) = amplitude_.logDeriv(ampDatas[n]);
		}

		MatrixType phaseLogDers(nSmp, numParams);
		torch::Tensor z = torch::zeros({nSmp,1}, torch::dtype(torch::kFloat64));
		auto z_a = z.accessor<double,2>();
		for(int n = 0; n < nSmp; ++n)
		{
			z_a[n][0] = 1.0;
			phaseNet_->zero_grad();
			phaseRes_t.backward(z, true);

			int s = 0;
			for(const auto p : parameters)
			{
				torch::Tensor g = torch::flatten(p.grad().to(torch::kCPU));
				g = g.contiguous();
				phaseLogDers.block(n, s, 1, g.numel()) = 
					Eigen::Map<const RowVectorType>(g.data_ptr<ScalarType>(), g.numel());
				s += g.numel();
			}
			z_a[n][0] = 0.0;
		}
		phaseLogDers *= M_PI;
		return std::make_tuple(coeffs, ampLogDers, phaseLogDers);
	}

	inline uint32_t getDimAmp() const
	{
		return amplitude_.getDim();
	}

	inline uint32_t getDimPhase() const
	{
		uint32_t n = 0;
		for(const auto p: phaseNet_->parameters())
		{
			n += p.numel();
		}
		return n;
	}

	inline uint32_t getDim() const
	{
		return getDimAmp() + getDimPhase();
	}

	std::complex<ScalarType> logCoeff(const AmpDataType& data) const
	{
		CxScalarType res{};
		res.real = amplitude_.logCoeff(data)/2.0;
		res.imag = M_PI*phaseForward(std::get<0>(data));
		return res;
	}

	ScalarType phaseForward(const Eigen::VectorXi& sigma) const
	{
		ScalarType res{};
		Eigen::VectorXd s = sigma.cast<ScalarType>();
		torch::Tensor s_t = torch::from_blob(s.data(), {1,1,N_}, opts_.requires_grad(false));
		torch::Tensor phaseRes_t = phaseNet_->forward(s_t);
		return phaseRes_t.item();
	}

	CxVectorType logCoeffs(const std::vector<AmpDataType>& ampData)
	{
		auto nSmp = ampData.size();
		CxVectorType res{};
#pragma omp parallel for schedule(static, 8)
		for(uint32_t idx = 0; idx < ampData.size(); idx++)
		{
			res[idx].real = amplitude_.logCoeff(ampData[idx])/2.0;
		}

		MatrixType sigmas(N_, ampData.size());
#pragma omp parallel for schedule(static, 8)
		for(uint32_t i = 0; i < ampData.size(); i++)
		{
			sigmas.col(i) = std::get<0>(ampData[i]);
		}

		torch::Tensor sigmas_t = torch::from_blob(sigmas.data(), {nSmp, 1, N_}, opts_.requires_grad(false));
		torch::Tensor phaseRes_t = phaseNet_->forward(sigmas_t);
		phaseRes_t = phaseRes_t.to(torch::kCPU);
		auto phaseRes_a = phaseRes_t.accessor<double,2>();

		for(int i = 0; i < phaseRes_a.size(0); i++) 
		{
			res[i].imag = M_PI*phaseRes_a[i][0];
		}
		return res;
	}

	template<typename RandomEngine>
	void initializeAmplitudeRandom(RandomEngine&& re, double sigma)
	{
		amplitude_.initializeRandom(re, sigma);
	}

	inline AmpDataType makeAmpData(const Eigen::Ref<const Eigen::VectorXi>& sigma) const
	{
		return amplitude_.makeData(sigma);
	}

	CxScalarType coeff(const AmpDataType& t) const
	{
		constexpr std::complex<double> I(0.,1.);
		return std::sqrt(amplitude_.coeff(t))*std::exp(I*M_PI*phaseForward(std::get<0>(t)));
	}

	uint32_t getN() const
	{
		return N_;
	}

	void updateParamsAmp(const VectorConstRefType& m)
	{
		assert(m.size() == getDimAmp());
		assert(m.innerStride() == 1);
		amplitude_.updateParams(m.head(amplitude_.getDim()));
	}

	void updateParamsPhase(const VectorConstRefType& m)
	{
		assert(m.size() == getDimPhase());
		assert(m.innerStride() == 1);
		int s = 0;
		for(auto p: phaseNet_->parameters())
		{
			torch::Tensor toUpdate = torch::from_blob(m.data(), p.sizes(), opts_);
			torch::NoGradGuard guard;
			p.add_(toUpdate);
		}
	}


	void updateParams(const VectorConstRefType& m)
	{
		assert(m.size() == getDim());
		amplitude_.updateParams(m.head(amplitude_.getDim()));
	}
};

template<class Net>
typename AmplitudePhase<Net>::CxVectorType 
getPsi(const AmplitudePhase<Net>& qs, bool normalize)
{
	const auto n = qs.getN();
	typename AmplitudePhase<Net>::CxVectorType psi(1<<n);
#pragma omp parallel for schedule(static,8)
	for(uint32_t i = 0; i < (1u<<n); i++)
	{
		auto s = toSigma(n, i);
		psi(i) = qs.coeff(qs.makeAmpData(s));
	}
	if(normalize)
		return psi.normalized();
	else
		return psi;
}

template<class Net>
typename AmplitudePhase<Net>::CxVectorType 
getPsi(const AmplitudePhase<Net>& qs, const std::vector<uint32_t>& basis, bool normalize)
{
	const auto n = qs.getN();
	typename AmplitudePhase<Net>::CxVectorType psi(basis.size());
#pragma omp parallel for schedule(static,8)
	for(uint32_t i = 0; i < basis.size(); i++)
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
