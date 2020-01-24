#ifndef YANNQ_TESTS_LAYERHELPER_HPP
#define YANNQ_TESTS_LAYERHELPER_HPP
#include <Machines/layers/AbstractLayer.hpp>
#include <Eigen/Dense>
#include <sstream>
#include "catch.hpp"
template<typename T>
typename yannq::AbstractLayer<T>::MatrixType 
ndiff_in(yannq::AbstractLayer<T>& layer, const typename yannq::AbstractLayer<T>::VectorType& input, const int outSize)
{
	using VectorT = typename yannq::AbstractLayer<T>::VectorType;
	using MatrixT = typename yannq::AbstractLayer<T>::MatrixType;
	MatrixT mat(input.size(), outSize);
	VectorT output1(outSize);
	VectorT output2(outSize);
	const double h = 1e-4;

	VectorT inH = input;
	for(int i = 0; i < input.size(); i++)
	{
		inH(i) += h;
		layer.forward(inH,output1);
		inH(i) = input(i) - h;
		layer.forward(inH,output2);
		mat.row(i) = (output1 - output2)/(2*h);
		inH(i) = input(i);
	}
	return mat;
}
template<typename T>
typename yannq::AbstractLayer<T>::MatrixType 
ndiff_weight(yannq::AbstractLayer<T>& layer, const typename yannq::AbstractLayer<T>::VectorType& input, const int outSize)
{
	using VectorT = typename yannq::AbstractLayer<T>::VectorType;
	using MatrixT = typename yannq::AbstractLayer<T>::MatrixType;
	MatrixT mat(layer.paramDim(), outSize);
	VectorT output1(outSize);
	VectorT output2(outSize);
	const double h = 1e-4;

	VectorT wH = layer.getParams();
	for(int i = 0; i < wH.size(); i++)
	{
		const auto val = wH(i);

		wH(i) = val + h;
		layer.setParams(wH);
		layer.forward(input,output1);

		wH(i) = val - h;
		layer.setParams(wH);
		layer.forward(input,output2);

		mat.row(i) = (output1 - output2)/(2*h);
		wH(i) = val;
	}
	return mat;
}


#endif//YANNQ_TESTS_LAYERHELPER_HPP
