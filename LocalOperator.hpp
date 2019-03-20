#ifndef CY_LOCAL_OPERATER_HPP
#define CY_LOCAL_OPERATER_HPP
#include "SampingResult.hpp"
namespace nnqs
{
template<typename T, class LocalOperatorT>
class LOState
{
private:
	NNQS<T>* qs_;
	LocalOperatorT op_;
	StateValue<T> st_;


	mutable bool cached_;
	mutable T denom_;
	mutable Eigen::MatrixXd ratio2Cache_;

	void calcDenom()
	{
		T s = op_(sigma_); //Id term
		for(int k = 0; k < qs_->getN(); k++)
		{
			s += op_(sigma_,k)*st_.ratio(k);
		}
		denom_ = s;
	}

	void calcRatio2()
	{
		const int n = qs_->getN();
		const int m = qs_->getM();
		for(int a = 0; a < n-1; a++)
		{
			for(int b = a + 1; b < n; b++)
			{
				T p = exp(-2.0*qs_->A(i)*st_.sigmaAt(i)-2.0*qs_->A(j)*st_.sigmaAt(j));
#pragma omp parallel for reduction(*:p) schedule(static,8)
				for(int j = 0; j < m; j++)
				{
					p *= cosh(st_.thetaAt(j)-2.0*st_.sigmaAt(a)*qs_->W(j,a)
							-2.0*st_.sigmaAt(b)*qs_->W(j,b))/cosh(st_.thetaAt(j));
				}
				ratio2Cache_.coeffRef(a,b) = p;
			}
		}
	}

	void calcCaches()
	{
		calcDenom();
		calcRatio2();
	}

public:
	template<typename SigmaT, typename ThetaT, typename LocalOperatorT>
	LOState(const NNQS<T>* qs, const LocalOperatorT& op, SigmaT&& sigma, ThetaT&& theta)
		: qs_(qs), op_(op), st_(qs, std::forward<SigmaT>(sigma), std::forward<ThetaT>(theta)),
		cached_(false), ratio2Cache_(qs->getN(), qs->getN())
	{
	}

	T ratio(int k) const
	{
		using std::min;
		using std::max;
		if(!cached)
		{
			calcRatios();
			cached = true;
		}

		T res = 0;
		VectorXi sp = st_.getSigma();
		sp.coeffRef(k) *= -1;
		for(auto elt: op_(sp))
		{
			int a = min(k, elt.second);
			int b = max(k, elt.second);
			res += elt.first*ratio2Cache_.coeff(a,b);
		}
		res /= denom;
		return res;
	}

	void flip(int k)
	{
		st_.flip(k);
		cached = false;
	}
};
}
#endif//CY_LOCAL_OPERATER_HPP
