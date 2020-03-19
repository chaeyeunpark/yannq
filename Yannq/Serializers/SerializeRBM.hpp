#ifndef YANNQ_SEREALIZAERS_SERIALIZERBM_HPP
#define YANNQ_SEREALIZAERS_SERIALIZERBM_HPP

#include <cereal/access.hpp> 
#include <cereal/types/memory.hpp>

#include "Machines/RBM.hpp"

CEREAL_CLASS_VERSION(yannq::RBM<float>, 1);
CEREAL_CLASS_VERSION(yannq::RBM<std::complex<float>>, 1);

CEREAL_CLASS_VERSION(yannq::RBM<double>, 1);
CEREAL_CLASS_VERSION(yannq::RBM<std::complex<double>>, 1);

CEREAL_CLASS_VERSION(yannq::RBM<long double>, 1);
CEREAL_CLASS_VERSION(yannq::RBM<std::complex<long double>>, 1);

namespace cereal
{
template<class Archive, typename T>
void save(Archive & ar, const yannq::RBM<T>& m, uint32_t const version)
{ 
	bool useBias = m.useBias();
	ar(useBias);
	ar(m.getN(),m.getM());
	ar(m.getW());
	if(!useBias)
		return ;
	ar(m.getA(),m.getB());
}

template<class Archive, typename T>
void load(Archive & ar, yannq::RBM<T>& m, uint32_t const version)
{ 
	bool useBias;
	ar(useBias);

	m.setUseBias(useBias);

	int N, M;
	ar(N, M);
	m.resize(N, M);

	typename yannq::RBM<T>::MatrixType W;
	ar(W);
	m.setW(W);
	if(!useBias)
		return ;
	typename yannq::RBM<T>::VectorType A, B;
	ar(A, B);
	m.setA(A);
	m.setB(B);
}

template <typename T>
struct LoadAndConstruct<yannq::RBM<T> >
{
	template<class Archive>
	static void load_and_construct(Archive& ar, cereal::construct<yannq::RBM<T> >& construct,  uint32_t const version)
	{
		bool useBias;
		ar(useBias);

		int n,m;
		ar(n, m);

		construct(n, m, useBias);
		
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W;
		ar(W);
		construct->setW(W);

		if(!useBias)
			return ;
		Eigen::Matrix<T, Eigen::Dynamic, 1> A;
		Eigen::Matrix<T, Eigen::Dynamic, 1> B;
		ar(A, B);

		construct->setA(A);
		construct->setB(B);
	}
};
}//namespace cereal

#endif//YANNQ_SEREALIZAERS_SERIALIZERBM_HPP
