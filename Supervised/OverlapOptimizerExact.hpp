#ifndef YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
#define YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
class OverlapOptimizerExact
{
private:
	int N_;
	Eigen::VectorXcd traget_;

public:

	explicit OverlapOptimizerExact(const Eigen::VectorXcd& target)
		: N_(qs.getN()), target_(target)
	{
	}
	Eigen::VectorXd gradientExact(const Machine& rbm) const
	{
		using std::conj;
		using Vector = Machine::Vector;
		Vector res(rbm.getDim());
		Vector r1(rbm.getDim());
		res.setZero();
		r1.setZero();

		Vector psi = getPsi(rbm, false);
#pragma omp parallel
		{
			Vector resLocal(rbm.getDim());
			Vector r1Local(rbm.getDim());
			resLocal.setZero();
			r1Local.setZero();

#pragma omp for schedule(static,8)
			for(uint32_t n = 0; n < (1u<<N_); n++)
			{
				auto s = yannq::toSigma(N_, n);
				auto der = rbm.logDeriv(std::make_tuple(s,rbm.calcTheta(s)));
				resLocal += der.conjugate()*std::norm(psi(n));
				r1Local += conj(psi(n))*target_(n)*der.conjugate();
			}
#pragma omp critical
			{
				res += resLocal;
				r1 += r1Local;
			}
		}
		std::complex<double> r = psi.adjoint() * target_;
		res -= r1/r;

		Eigen::VectorXd rr = Eigen::Map<Eigen::VectorXd>((double*)res.data(), 2*rbm.getDim(), 1);
		return rr;
	}

};
#endif//YANNQ_SUPERVISED_OVERLAPOPTIMIZEREXACT_HPP
