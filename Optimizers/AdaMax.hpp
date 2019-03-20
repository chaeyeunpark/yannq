#ifndef CY_ADAMAX_HPP
#define CY_ADAMAX_HPP

namespace nnqs
{
template<typename T>
class AdaMax
{
private:
	double alpha_;
	double beta1_;
	double beta2_;
	

	double u_;
	int t_;
	typename NNQS<T>::Vector m_;

public:
	using Vector = typename NNQS<T>::Vector;
	using ConstVector=typename NNQS<T>::ConstVector;
	AdaMax(std::size_t dim, double alpha, double beta1, double beta2)
		: alpha_(alpha), beta1_(beta1), beta2_(beta2), t_(0)
	{
		u_ = 0;
		m_ = Vector::Zero(dim);
	}

	Vector getUpdate(const typename NNQS<T>::Vector& v)
	{
		using std::pow;
		++t_;
		m_ *= beta1_;
		m_ += (1.0-beta1_)*v;
		u_ = std::max(beta2_*u_, v.norm());
		return -(alpha_/(1-pow(beta1_,t_)))*m_/u_;
	}

};
}//namespace nnqs

#endif//CY_ADAMAX_HPP
