#pragma once

#include <cereal/access.hpp> 
#include <cereal/types/memory.hpp>

#include "Machines/CorrelatedRBM.hpp"

CEREAL_CLASS_VERSION(yannq::CorrelatedRBM<float>, 1);
CEREAL_CLASS_VERSION(yannq::CorrelatedRBM<std::complex<float>>, 1);

CEREAL_CLASS_VERSION(yannq::CorrelatedRBM<double>, 1);
CEREAL_CLASS_VERSION(yannq::CorrelatedRBM<std::complex<double>>, 1);

CEREAL_CLASS_VERSION(yannq::CorrelatedRBM<long double>, 1);
CEREAL_CLASS_VERSION(yannq::CorrelatedRBM<std::complex<long double>>, 1);

namespace cereal
{
template<class Archive, typename T>
void save(Archive & ar, const yannq::CorrelatedRBM<T>& m, uint32_t const /*version*/)
{ 
	bool useBias = m.useBias();
	ar(useBias);
	ar(m.getN(),m.getM());

	ar(m.getCorrel());

	ar(m.getW());
	if(!useBias)
		return ;
	ar(m.getA(),m.getB());
}

template<class Archive, typename T>
void load(Archive & ar, yannq::CorrelatedRBM<T>& m, uint32_t const /*version*/)
{ 
	bool useBias;
	int N, M;
	ar(useBias);
	ar(N, M);

	m.resize(N, M);
	m.setUseBias(useBias);

	Eigen::SparseMatrix<int> correl;
	ar(correl);
	m.setCorrel(correl);

	typename yannq::CorrelatedRBM<T>::Matrix W;
	ar(W);
	m.setW(W);
	if(!useBias)
		return ;
	typename yannq::CorrelatedRBM<T>::Vector A, B;
	ar(A, B);
	m.setA(A);
	m.setB(B);
}
}//namespace cereal
