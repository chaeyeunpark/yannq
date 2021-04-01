#ifndef YANNQ_GROUNDSTATE_NGDEXACT_HPP
#define YANNQ_GROUNDSTATE_NGDEXACT_HPP
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <tbb/tbb.h>

#include "ED/ConstructSparseMat.hpp"
#include "Utilities/Utility.hpp"

#include "Machines/AmplitudePhase.hpp"

namespace yannq
{

class NGDExact
{
public:
	using Machine = AmplitudePhase;
	using RealScalar = typename Machine::RealScalar;
	using ComplexScalar = typename Machine::ComplexScalar;

	using RealMatrix = typename Machine::RealMatrix;
	using RealVector = typename Machine::RealVector;

	using ComplexMatrix = typename Machine::ComplexMatrix;
	using ComplexVector = typename Machine::ComplexVector;

private:
	const uint32_t n_;
	const AmplitudePhase& qs_;
	tbb::concurrent_vector<uint32_t> basis_;

	Eigen::SparseMatrix<RealScalar> ham_;

	RealMatrix deltasAmp_;
	RealMatrix deltasPhase_;

	RealMatrix deltasAmpPsis_;
	RealMatrix deltasPhasePsis_;

	RealVector olocAmp_;
	RealVector olocPhase_;

	RealVector grad_;

	RealScalar energy_;

	void constructDeltaAmp()
	{
		using Range = tbb::blocked_range<std::size_t>;
		const int N = qs_.getN();
		deltasAmp_.setZero(basis_.size(), qs_.getDimAmp());
		if(basis_.size() >= 32)
		{
			tbb::parallel_for(Range(std::size_t(0u), basis_.size(), 8),
				[&](const Range& r)
			{
				uint32_t start = r.begin();
				uint32_t end = r.end();
				RealMatrix tmp(end-start, qs_.getDimAmp());
				for(uint32_t l = 0; l < end-start; ++l)
				{
					tmp.row(l) = 
						qs_.logDerivAmp(qs_.makeAmpData(toSigma(N, basis_[l+start])));
				}
				deltasAmp_.block(start, 0, end-start, qs_.getDimAmp()) = tmp;
			}, tbb::simple_partitioner());
		}
		else
		{
			for(uint32_t k = 0; k < basis_.size(); k++)
			{
				deltasAmp_.row(k) = 
					qs_.logDerivAmp(qs_.makeAmpData(toSigma(N, basis_[k])));
			}
		}
	}

	void constructDeltaPhase()
	{
		using Range = tbb::blocked_range<std::size_t>;
		const int N = qs_.getN();
		deltasPhase_.setZero(basis_.size(), qs_.getDimPhase());
	
		if(basis_.size() >= 32)
		{
			tbb::parallel_for(Range(std::size_t(0u), basis_.size(), 8),
				[&](const Range& r)
			{
				uint32_t start = r.begin();
				uint32_t end = r.end();
				RealMatrix tmp(end-start, qs_.getDimPhase());
				for(uint32_t l = 0; l < end-start; ++l)
				{
					tmp.row(l) = 
						qs_.logDerivPhase(qs_.makePhaseData(toSigma(N, basis_[l+start])));
				}
				deltasPhase_.block(start, 0, end-start, qs_.getDimPhase()) = tmp;
			}, tbb::simple_partitioner());
		}
		else
		{
			for(uint32_t k = 0; k < basis_.size(); k++)
			{
				deltasPhase_.row(k) = 
					qs_.logDerivPhase(qs_.makePhaseData(toSigma(N, basis_[k])));
			}
		
		}
	}

public:

	RealScalar getEnergy() const
	{
		return energy_;
	}


	void constructExact()
	{
		ComplexVector st = getPsi(qs_, basis_, true);
		ComplexVector k = ham_*st;

		energy_ = std::real(ComplexScalar(st.adjoint()*k));

		constructDeltaAmp();
		constructDeltaPhase();
		
		deltasAmpPsis_ = st.cwiseAbs2().asDiagonal()*deltasAmp_; 
		olocAmp_ = deltasAmpPsis_.colwise().sum();

		deltasPhasePsis_ = st.cwiseAbs2().asDiagonal()*deltasPhase_; 
		olocPhase_ = deltasPhasePsis_.colwise().sum();
		
		k = st.conjugate().asDiagonal()*k;
		grad_.resize(qs_.getDim());
		grad_.head(qs_.getDimAmp()) = 
			2.0*deltasAmp_.transpose()*k.real();
		grad_.head(qs_.getDimAmp()) -= 2.0*energy_*olocAmp_;
		grad_.tail(qs_.getDimPhase()) 
			= 2.0*deltasPhase_.transpose()*k.imag();
		/*
		grad_.tail(qs_.getDimPhase()) 
			-= 2.0*energy_*olocPhase_;
			*/
	}

	RealScalar eloc() const
	{
		return energy_;
	}

	const RealVector& olocAmp() const&
	{
		return olocAmp_;
	}

	RealMatrix olocAmp() &&
	{
		return olocAmp_;
	}

	const RealVector& olocPhase() const&
	{
		return olocPhase_;
	}

	RealMatrix olocPhase() &&
	{
		return olocPhase_;
	}

	RealMatrix corrMatAmp() const
	{
		RealMatrix res = deltasAmp_.transpose()*deltasAmpPsis_;
		res -= olocAmp_*olocAmp_.transpose();
		return res;
	}

	RealMatrix corrMatPhase() const
	{
		RealMatrix res = deltasPhase_.transpose()*deltasPhasePsis_;
		res -= olocPhase_*olocPhase_.transpose();
		return res;
	}

	RealVector energyGrad() const
	{
		return grad_;
	}

	template<class Iterable, class ColFunc>
	NGDExact(const AmplitudePhase& qs, Iterable&& basis, ColFunc&& col)
	  : n_{qs.getN()}, qs_(qs)
	{
		tbb::parallel_do(basis.begin(), basis.end(), 
				[&](uint32_t elt)
		{
			basis_.emplace_back(elt);
		});
		tbb::parallel_sort(basis_.begin(), basis_.end());

		ham_ = edp::constructSubspaceMat(std::forward<ColFunc>(col), basis_);
	}
};
} //namespace yannq
#endif//YANNQ_GROUNDSTATE_NGDEXACT_HPP
