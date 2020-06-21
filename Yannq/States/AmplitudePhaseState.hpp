#ifndef YANNQ_STATES_RBMSTATE_HPP
#define YANNQ_STATES_RBMSTATE_HPP

#include "Machines/AmplitudePhase.hpp"
#include "Utilities/type_traits.hpp"
#include "Utilities/Utility.hpp"
#include "./RBMState.hpp"

namespace yannq
{

template<class Derived>
class AmplitudePhaseStateObj
{
protected:
	const AmplitudePhase& qs_;

public:
	using Scalar = typename AmplitudePhase::Scalar;
	using CxScalar = typename AmplitudePhase::CxScalar;

	using PhaseDataType = typename AmplitudePhase::PhaseDataType;

protected:
	AmplitudePhaseStateObj(const AmplitudePhase& qs) noexcept
		: qs_(qs)
	{
	}

public:

	const Eigen::VectorXi& getSigma() const
	{
		return static_cast<const Derived*>(this)->getSigma();
	}

	const auto& getAmplitudeState() const
	{
		return static_cast<const Derived*>(this)->getAmplitudeState();
	}

	auto& getAmplitudeState() 
	{
		return static_cast<const Derived*>(this)->getAmplitudeState();
	}

	const PhaseDataType& getPhaseData() const
	{
		return static_cast<Derived*>(this)->getPhaseData();
	}
	
	Scalar logRatioRe(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		return getAmplitudeState().logRatio(k)/2.0;
	}

	CxScalar logRatio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		CxScalar res;
		res.real(getAmplitudeState().logRatio(k)/2.0);
		Eigen::VectorXi flipped = getSigma();
		flipped(k) *= -1;
		res.imag( M_PI*( qs_.phaseForward(flipped) - 
					qs_.phaseForward(getPhaseData())));
		return res;
	}

	inline CxScalar ratio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		return std::exp(logRatio(k));
	}
	
	Scalar logRatioRe(int k, int l) const //calc psi(sigma ^ k ^ l)/psi(sigma)
	{
		return getAmplitudeState().logRatio(k,l)/2.0;
	}

	CxScalar logRatio(int k, int l) const //calc psi(sigma ^ k ^ l)/psi(sigma)
	{
		CxScalar res;
		res.real(getAmplitudeState().logRatio(k,l)/2.0);
		Eigen::VectorXi flipped = static_cast<Derived*>(this)->getSigma();
		flipped(k) *= -1;
		flipped(l) *= -1;
		res.imag( M_PI*( qs_.phaseForward(flipped) -
					qs_.phaseForward(getPhaseData())));
		return res;
	}

	inline CxScalar ratio(int k, int l) const
	{
		return std::exp(logRatio(k,l));
	}

	template<std::size_t N>
	Scalar logRatioRe(const std::array<int, N>& v) const
	{
		return getAmplitudeState().logRatio(v)/2.0;
	}

	template<std::size_t N>
	CxScalar logRatio(const std::array<int, N>& v) const
	{
		CxScalar res;
		res.real(getAmplitudeState().logRatio(v)/2.0);
		Eigen::VectorXi flipped = static_cast<Derived*>(this)->getSigma();
		for(int k : v)
		{
			flipped(k) *= -1;
		}
		res.imag( M_PI*( qs_.phaseForward(flipped) -
					qs_.phaseForward(getPhaseData())));
		return res;
	}

	const AmplitudePhase& getMachine() const
	{
		return qs_;
	}
};

struct AmplitudePhaseStateValue
	: public AmplitudePhaseStateObj<AmplitudePhaseStateValue>
{
private:
	RBMStateValue<AmplitudePhase::AmplitudeMachine> ampState_;
	AmplitudePhase::PhaseDataType phaseData_;

public:
	using AmplitudeDataType = AmplitudePhase::AmplitudeDataType;
	using PhaseDataType = AmplitudePhase::PhaseDataType;
	using AmplitudeStateType = RBMStateValue<AmplitudePhase::AmplitudeMachine>;
	using Vector = typename AmplitudePhase::Vector;
	using Scalar = AmplitudePhaseStateObj<AmplitudePhaseStateValue>::Scalar;
	using CxScalar = AmplitudePhaseStateObj<AmplitudePhaseStateValue>::CxScalar;

	AmplitudePhaseStateValue(const AmplitudePhase& qs, Eigen::VectorXi&& sigma) noexcept
		: AmplitudePhaseStateObj<AmplitudePhaseStateValue>(qs), 
		ampState_(qs.amplitudeMachine(), std::move(sigma))
	{
		phaseData_ = qs_.makePhaseData(ampState_.getSigma());
	}

	AmplitudePhaseStateValue(const AmplitudePhase& qs, const Eigen::VectorXi& sigma) noexcept
		: AmplitudePhaseStateObj<AmplitudePhaseStateValue>(qs), 
		ampState_(qs.amplitudeMachine(), sigma)
	{
		phaseData_ = qs_.makePhaseData(sigma);
	}

	AmplitudePhaseStateValue(const AmplitudePhaseStateValue& rhs) = default;
	AmplitudePhaseStateValue(AmplitudePhaseStateValue&& rhs) = default;

	AmplitudePhaseStateValue& operator=(const AmplitudePhaseStateValue& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		ampState_ = rhs.ampState_;
		phaseData_ = rhs.phaseData_;
		return *this;
	}

	AmplitudePhaseStateValue& operator=(AmplitudePhaseStateValue&& rhs) noexcept
	{
		assert(rhs.qs_ == this->qs_);
		ampState_ = std::move(rhs.ampState_);
		phaseData_ = std::move(rhs.phaseData_);
		return *this;
	}

	void setSigma(const Eigen::VectorXi& sigma)
	{
		ampState_.setSigma(sigma);
		phaseData_ = qs_.makePhaseData(sigma);
	}

	void setSigma(Eigen::VectorXi&& sigma)
	{
		ampState_.setSigma(std::move(sigma));
		phaseData_ = qs_.makePhaseData(ampState_.getSigma());
	}

	inline const Eigen::MatrixXi& getSigma() const&
	{
		return ampState_.getSigma();
	}

	inline Eigen::MatrixXi getSigma() &&
	{
		return ampState_.getSigma();
	}

	void flip(int k)
	{
		ampState_.flip(k);
		phaseData_ = qs_.makePhaseData(ampState_.getSigma());
	}
	
	template<std::size_t N>
	void flip(const std::array<int, N>& v)
	{
		ampState_.flip(v);
		phaseData_ = qs_.makePhaseData(ampState_.getSigma());
	}

	void flip(int k, int l)
	{
		ampState_.flip(k,l);
		phaseData_ = qs_.makePhaseData(ampState_.getSigma());
	}

	using AmplitudePhaseStateObj<AmplitudePhaseStateValue>::logRatio;
	CxScalar logRatio(const AmplitudePhaseStateValue& other)
	{
		CxScalar res;
		res.real(ampState_.logRatio(other)/2.0);
		res.imag(M_PI*(qs_.phaseForward(phaseData_) -
					qs_.phaseForward(other.getSigma())));
	}

	const PhaseDataType& getPhaseData() const&
	{
		return phaseData_;
	}

	const AmplitudeStateType& getAmplitudeState() const&
	{
		return ampState_;
	}

	AmplitudeStateType getAmplitudeState() &&
	{
		return ampState_;
	}

	std::tuple<AmplitudeDataType, PhaseDataType> makeData() const
	{
		return std::make_tuple(ampState_.data(), phaseData_);
	}
};


class AmplitudePhaseStateRef
	: public AmplitudePhaseStateObj<AmplitudePhaseStateRef>
{
public:
	using Vector = typename Machine::Vector;
private:
	const RBMStateRef<AmplitudePhase::AmplitudeMachine> ampState_;
	const AmplitudePhase::PhaseDataType& phaseData_;
public:
	using AmplitudeDataType = AmplitudePhase::AmplitudeDataType;
	using PhaseDataType = AmplitudePhase::PhaseDataType;
	using AmplitudeStateType = RBMStateRef<AmplitudePhase::AmplitudeMachine>;
	using Vector = typename AmplitudePhase::Vector;
	using Scalar = AmplitudePhaseStateObj<AmplitudePhaseStateValue>::Scalar;
	using CxScalar = AmplitudePhaseStateObj<AmplitudePhaseStateValue>::CxScalar;

	AmplitudePhaseStateRef(const AmplitudePhase& qs, const AmplitudeDataType& ampData, const PhaseDataType& phaseData) noexcept
		: AmplitudePhaseStateObj<AmplitudePhaseStateRef>(qs), 
		ampState_(qs.amplitudeMachine(), ampData), phaseData_(phaseData)
	{
	}

	inline int sigmaAt(int i) const
	{
		return ampState_.getSigma()(i);
	}

	Eigen::VectorXi getSigma() const
	{
		return ampState_.getSigma();
	}

	const AmplitudeStateType& getAmplitudeState() const
	{
		return ampState_;
	}
	
};

template <>
struct is_reference_state_type<AmplitudePhaseStateRef>: public std::true_type {};

} //namespace yannq
#endif//YANNQ_STATES_RBMSTATE_HPP
