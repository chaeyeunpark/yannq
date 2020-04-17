#ifndef YANNQ_HYPERCUBECONVLAYER_HH
#define YANNQ_HYPERCUBECONVLAYER_HH

#include <time.h>
#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "AbstractLayer.hpp"

#include <Utilities/Utility.hpp>
#include <Utilities/Exceptions.hpp>

namespace yannq {
/** Convolutional layer with spin 1/2 hidden units.
 Important: In order for this to work correctly, VectorType and MatrixType must
 be column major.
 */
template<typename T>
class Conv1D : public AbstractLayer<T> {
	static_assert(!AbstractLayer<T>::MatrixType::IsRowMajor, "MatrixType must be column-major");

public:
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

private:
	const bool useBias_;  // boolean to turn or off bias

	const uint32_t inChannels_;   // number of input channels
	const uint32_t outChannels_;  // number of output channels

	const uint32_t kernelSize_;
	const uint32_t stride_;         // convolution stride
	
	const uint32_t npar_;          // number of parameters in layer

	MatrixType kernel_;  // Weight parameters, W((inChannels_ * kernelSize)x(outChannels))
	VectorType bias_;     // Bias parameters, b(outChannels)

	static uint32_t numParams(bool useBias, const uint32_t inChannels,
			const uint32_t outChannels, const uint32_t kernelSize)
	{
		uint32_t np = inChannels * outChannels * kernelSize;
		if(useBias)
			np += outChannels;
		return np;
	}

public:
	/// Constructor
	Conv1D(	const uint32_t inChannels, const uint32_t outChannels,
			const uint32_t kernelSize, const uint32_t stride = 1,
			const bool useBias = false)
		: useBias_(useBias), inChannels_(inChannels), outChannels_(outChannels),
		kernelSize_(kernelSize), stride_(stride),
		npar_(numParams(useBias, inChannels, outChannels, kernelSize)),
		kernel_(inChannels*kernelSize, outChannels), bias_(outChannels)
	{
	}

	Conv1D(const Conv1D& rhs) = default;
	Conv1D(Conv1D&& rhs) = default;

	Conv1D& operator=(const Conv1D& rhs) = default;
	Conv1D& operator=(Conv1D&& rhs) = default;

	bool operator==(const Conv1D& rhs) const
	{
		if(useBias_ != rhs.useBias_)
			return false;

		bool res = (inChannels_ == rhs.inChannels_) && 
				(outChannels_ == rhs.outChannels_) &&
				(kernelSize_ == rhs.kernelSize_) &&
				(stride_ == rhs.stride_) &&
				(kernel_ == rhs.kernel_);

		if(!useBias_)
			return res;
		else
			return res && (bias_ == rhs.bias_);
	}

	std::string name() const override { return "Convolutional 1D Layer"; }

	template<class RandomEngine>
	void randomizeParams(RandomEngine&& re, double sigma)
	{
		setParams(randomVector<T>(std::forward<RandomEngine>(re), sigma, npar_));
	}

	uint32_t paramDim() const override { return npar_; }
	uint32_t outputDim(uint32_t inputDim) const override {
		return (inputDim / stride_ / inChannels_) * outChannels_; 
	}

	VectorType getParams() const override 
	{
		VectorType pars(npar_);
		if(useBias_)
		{
			pars.head(outChannels_) = bias_;
			pars.segment(outChannels_, kernel_.size()) = 
				Eigen::Map<const VectorType>(kernel_.data(), kernel_.rows()*kernel_.cols());
		}
		else
		{
			pars = Eigen::Map<const VectorType>(kernel_.data(), kernel_.rows()*kernel_.cols());
		}
		return pars;
	}

	void setParams(VectorConstRefType pars) override 
	{
		if(useBias_)
		{
			bias_ = pars.head(outChannels_);
			kernel_ = Eigen::Map<const MatrixType>(pars.data() + outChannels_, kernel_.rows(), kernel_.cols());
		}
		else
		{
			kernel_ = Eigen::Map<const MatrixType>(pars.data(), kernel_.rows(), kernel_.cols());
		}
	}

	void updateParams(VectorConstRefType ups) override
	{
		if(useBias_)
		{
			bias_ += ups.head(outChannels_);
			kernel_ += Eigen::Map<const MatrixType>(ups.data() + outChannels_, kernel_.rows(), kernel_.cols());
		}
		else
		{
			kernel_ += Eigen::Map<const MatrixType>(ups.data(), kernel_.rows(), kernel_.cols());
		}
	}

	/**
	 * Feedforward
	 * @input: inChannels*size
	 * @output: outChannels*size
	 */
	void forward(const VectorConstRefType& input, VectorRefType output) override 
	{
		assert(input.size() % inChannels_ == 0);
		uint32_t inSize = input.size() / inChannels_;
		uint32_t outSize = inSize / stride_;

		output.setZero();

		// y = Wx+b
		for (uint32_t oc = 0; oc < outChannels_; oc++)
		for (uint32_t r = 0; r < outSize; r ++) 
		{
			for (uint32_t ic = 0; ic < inChannels_; ic++)
			for (uint32_t ki = 0; ki < kernelSize_; ki++)
			{
				output(r + oc*outSize) += kernel_(ki + ic*kernelSize_, oc)
					*input(((r*stride_+ki-kernelSize_/2+inSize)%inSize) + ic*inSize);
			}
		}

		if (useBias_) {
			for (uint32_t oc = 0; oc < outChannels_; ++oc) {
				for (uint32_t i = 0; i < outSize; ++i) {
					output(i + oc*outSize) += bias_(oc);
				}
			}
		}
	}

	void backprop(const VectorConstRefType& prev_layer_output,
			const VectorConstRefType& this_layer_output,
			const VectorConstRefType& dout, 
			VectorRefType din,
			VectorRefType der) override
	{
		assert(prev_layer_output.size() % inChannels_ == 0);
		uint32_t inSize = prev_layer_output.size() / inChannels_;
		uint32_t outSize = inSize / stride_;

		din.resize(inSize*inChannels_);
		din.setZero();
		der.setZero();

		// propagate delta to prev-layer
		for (uint32_t oc = 0; oc < outChannels_; oc++)
		for (uint32_t r = 0; r < outSize; r ++) 
		for (uint32_t ic = 0; ic < inChannels_; ic++)
		for (uint32_t ki = 0; ki < kernelSize_; ki++)
		{
			din(((r*stride_+ki-kernelSize_/2+inSize)%inSize) + ic*inSize) 
				+= kernel_(ki + ic*kernelSize_, oc)*dout[r + oc*outSize];
		}
		
		uint32_t k = 0;
		if(useBias_)// accumulate bias difference
		{
			for (uint32_t oc = 0; oc < outChannels_; oc++)
			for (uint32_t r = 0; r < outSize; r ++) 
			{
				der(oc) += dout(r + oc*outSize);
			}
			k += outChannels_;
		}

		MatrixType dw(inChannels_*kernelSize_, outChannels_);
		dw.setZero();

		// accumulate weight difference
		for (uint32_t oc = 0; oc < outChannels_; oc++)
		for (uint32_t r = 0; r < outSize; r ++) 
		{
			for (uint32_t ic = 0; ic < inChannels_; ic++)
			for (uint32_t ki = 0; ki < kernelSize_; ki++)
			{
				dw(ki + ic*kernelSize_, oc) += 
					prev_layer_output(((r*stride_+ki-kernelSize_/2+inSize)%inSize) + ic*inSize)*
					dout[r + oc*outSize];
			}
		}
		dw.resize(dw.rows()*dw.cols(),1);
		der.segment(k, kernel_.rows()*kernel_.cols()) = std::move(dw);
	}

	uint32_t fanIn() override
	{
		return inChannels_*kernelSize_;
	}
	uint32_t fanOut() override
	{
		return outChannels_*kernelSize_;
	}

	nlohmann::json desc() const override {
		nlohmann::json layerpar;
		layerpar["name"] = name();
		layerpar["use_bias"] = useBias_;
		layerpar["input_channels"] = inChannels_;
		layerpar["output_channels"] = outChannels_;
		layerpar["kernel_size"] = kernelSize_;
		layerpar["stride"] = stride_;
		return layerpar;
	}
};
}// namespace yannq

#endif
