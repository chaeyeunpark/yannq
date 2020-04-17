#ifndef YANNQ_MACHINES_AMPLITUDEPHASE_HPP
#define YANNQ_MACHINES_AMPLITUDEPHASE_HPP
#include <tbb/tbb.h>

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

	RBM<ScalarType> amplitude_;
	FeedForward<ScalarType> phase_;

public:
	using AmplitudeMachine = RBM<ScalarType>;
	using ReScalarType = ScalarType;
	using CxScalarType = std::complex<double>;
	using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorType = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

	using ReMatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using ReVectorType = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

	using CxMatrixType = Eigen::Matrix<CxScalarType, Eigen::Dynamic, Eigen::Dynamic>;
	using CxVectorType = Eigen::Matrix<CxScalarType, Eigen::Dynamic, 1>;
	using AmplitudeDataType = std::tuple<Eigen::VectorXi, VectorType>;
	using PhaseDataType = std::vector<VectorType>;

	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

	AmplitudePhase(uint32_t N, uint32_t M, bool useBias, FeedForward<ScalarType>&& phase)
		: N_{N}, amplitude_(N, M, useBias), phase_(std::move(phase))
	{
	}

	VectorType logDerivAmp(const AmplitudeDataType& data) const
	{
		return amplitude_.logDeriv(data)/2.0;
	}

	VectorType logDerivPhase(const PhaseDataType& data) const
	{
		return (M_PI)*phase_.backward(data);
	}

	CxVectorType logDeriv(const std::pair<AmplitudeDataType,PhaseDataType>& data) const
	{
		constexpr CxScalarType I(0., 1.);
		CxVectorType res(getDim());
		res.head(amplitude_.getDim()) = logDerivAmp(data.first);
		res.tail(phase_.getDim()) = I*logDerivPhase(data.second);
		return res;
	}

	inline uint32_t getDimAmp() const
	{
		return amplitude_.getDim();
	}

	inline uint32_t getDimPhase() const
	{
		return phase_.getDim();
	}

	inline uint32_t getDim() const
	{
		return amplitude_.getDim() + phase_.getDim();
	}

	const AmplitudeMachine& amplitudeMachine() const
	{
		return amplitude_;
	}

	template<typename RandomEngine>
	void initializeAmplitudeRandom(RandomEngine& re, double sigma)
	{
		amplitude_.initializeRandom(re, sigma);
	}

	template<typename RandomEngine>
	void initializeRandom(RandomEngine& re, double sigma)
	{
		initializeAmplitudeRandom(re, sigma);
		//phase_.initializeRandom(re, InitializationMode::Xavier);
		phase_.initializeRandom(re, InitializationMode::LeCun);
	}


	AmplitudeDataType makeAmpData(const Eigen::VectorXi& sigma) const
	{
		return amplitude_.makeData(sigma);
	}
	
	PhaseDataType makePhaseData(const Eigen::VectorXi& sigma) const
	{
		return phase_.makeData(sigma);
	}

	std::pair<AmplitudeDataType,PhaseDataType> makeData(const Eigen::VectorXi& sigma) const
	{
		return std::make_pair(amplitude_.makeData(sigma), phase_.makeData(sigma));
	}

	ScalarType phaseForward(const PhaseDataType& phaseData) const
	{
		return phase_.forward(phaseData);
	}

	ScalarType phaseForward(const Eigen::VectorXi& sigma) const
	{
		return phase_.forward(sigma);
	}

	CxScalarType coeff(const AmplitudeDataType& t) const
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

	nlohmann::json dsec() const
	{
		nlohmann::json res;
		res["amplitude"] = amplitude_.desc();
		res["phase"] = phase_.desc();
		return res;
	}
};

AmplitudePhase::CxVectorType getPsi(const AmplitudePhase& qs, bool normalize)
{
	const uint32_t n = qs.getN();
	AmplitudePhase::CxVectorType psi(1<<n);
	tbb::parallel_for(uint32_t(0u), (1u << n), 
		[n, &qs, &psi](uint32_t idx)
	{
		auto s = toSigma(n, idx);
		psi(idx) = qs.coeff(qs.makeAmpData(s));
	});
	if(normalize)
		psi.normalize();
	return psi;
}

template<typename Iterable> //Iterable must be random access iterable
AmplitudePhase::CxVectorType getPsi(const AmplitudePhase& qs, Iterable&& basis, bool normalize)
{
	const uint32_t n = qs.getN();
	AmplitudePhase::CxVectorType psi(basis.size());

	tbb::parallel_for(std::size_t(0u), basis.size(),
		[n, &qs, &psi, &basis](std::size_t idx)
	{
		auto s = toSigma(n, basis[idx]);
		psi(idx) = qs.coeff(qs.makeAmpData(s));
	});
	if(normalize)
		psi.normalize();
	return psi;
}
}//namespace yannq
#endif//YANNQ_MACHINES_AMPLITUDEPHASE_HPP
