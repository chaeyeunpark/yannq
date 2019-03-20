#ifndef CY_NQS_SAMPLING_HPP
#define CY_NQS_SAMPLING_HPP
#include <cmath>
#include <armadillo>
#include "NNQS.hpp"
class Sampling
{
private:
	const int n_;
	const int m_;
	const NNQS* qs_;

	arma::ivec currSigma_;
	arma::vec theta_; //length m

public:
	inline int sigmaAt(int i) const
	{
		return currSigma_(i);
	}
	inline double thetaAt(int j) const
	{
		return theta_.at(j);
	}
	inline const arma::ivec& getSigma() const
	{
		return currSigma_;
	}
	inline const arma::vec& getTheta() const
	{
		return theta_;
	}
	template<typename RandomEngine>
	Sampling(NNQS* qs, RandomEngine& re)
		: n_(qs->n_), m_(qs->m_), qs_(qs), currSigma_(n_)
	{
		randomizeSigma(re);

	}
	template <typename RandomEngine>
	void randomizeSigma(RandomEngine& re)
	{
		arma::ivec sigma(n_);
		std::uniform_int_distribution<> uid(0, 1);
		//randomly initialize currSigma
		for(int i = 0; i < n_; i++)
		{
			sigma.at(i) = -2*uid(re)+1;
		}
		setCurrSigma(std::move(sigma));
	}

	void setCurrSigma(arma::ivec sigma)
	{
		currSigma_.swap(sigma);
		theta_ = qs_->calcTheta(currSigma_);
	}

	double ratio(int k) const //calc psi(sigma ^ k) / psi(sigma)
	{
		using std::exp;
		using std::cosh;
		double p = exp(-2.0*(qs_->a_).at(k)*currSigma_.at(k));
#pragma omp parallel for reduction(*:p) schedule(static,8)
		for(int j = 0; j < m_; j++)
		{
			double s = cosh(theta_.at(j)-2*currSigma_(k)*(qs_->W(j,k)))/cosh(theta_.at(j));
			p *= s;
		}

		return p;
	}

	void flip(int k)
	{
#pragma omp parallel for schedule(static,8)
		for(int j = 0; j < m_; j++)
		{
			theta_(j) -= 2*currSigma_(k)*(qs_->W(j,k));
		}
		currSigma_.at(k) *= -1;
	}
	
	//H = \sum J sigma^z sigma^z + h \sigma^x
	template<typename RandomEngine, class Op>
	void Calc(RandomEngine& re, Op& op, int n_sweeps, double therm_factor = 0.1, int sweep_factor = 1.0)
	{
		using std::pow;
		using std::abs;
		std::uniform_real_distribution<> urd(0.0,1.0);
		std::uniform_int_distribution<> uid(0, n_-1);

		int ntherm = int(therm_factor*n_sweeps)*int(sweep_factor*n_);

		//Thermalizing phase
		for(int n = 0; n < ntherm; n++)
		{
			int toFlip = uid(re);
			double p = std::min(1.0,pow(ratio(toFlip),2));
			double u = urd(re);
			if(u < p)//accept
			{
				flip(toFlip);
			}
		}

		for(int ll = 0; ll < n_sweeps; ll++)
		{
			//sweep
			for(int i = 0; i < int(sweep_factor*n_); i++)
			{
				int toFlip = uid(re);
				double p = std::min(1.0,pow(ratio(toFlip),2));
				double u = urd(re);
				if(u < p)//accept
				{
					flip(toFlip);
				}
			}
			op(*this);
		}
		op.finishIter();
	}
};
#endif//CY_NQS_SAMPLING_HPP
