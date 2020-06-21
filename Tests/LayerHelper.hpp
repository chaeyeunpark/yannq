#ifndef YANNQ_TESTS_LAYERHELPER_HPP
#define YANNQ_TESTS_LAYERHELPER_HPP
#include <Eigen/Dense>
#include <sstream>

#include <Machines/layers/AbstractLayer.hpp>
#include <Utilities/type_traits.hpp>

template<typename T>
typename yannq::AbstractLayer<T>::Matrix
ndiff_in(yannq::AbstractLayer<T>& layer, const typename yannq::AbstractLayer<T>::Vector& input, const int outSize)
{
	using Vector = typename yannq::AbstractLayer<T>::Vector;
	using Matrix = typename yannq::AbstractLayer<T>::Matrix;
	Matrix mat(input.size(), outSize);
	Vector output1(outSize);
	Vector output2(outSize);
	const typename yannq::remove_complex<T>::type h = 1e-5;

	Vector inH = input;
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
typename yannq::AbstractLayer<T>::Matrix
ndiff_weight(yannq::AbstractLayer<T>& layer, const typename yannq::AbstractLayer<T>::Vector& input, const int outSize)
{
	using Vector = typename yannq::AbstractLayer<T>::Vector;
	using Matrix = typename yannq::AbstractLayer<T>::Matrix;
	Matrix mat(layer.paramDim(), outSize);
	Vector output1(outSize);
	Vector output2(outSize);
	const typename yannq::remove_complex<T>::type h = 1e-5;

	Vector wH = layer.getParams();
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
