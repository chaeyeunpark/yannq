#pragma once
#include "AbstractLayer.hpp"
#include "Utilities/type_traits.hpp"
namespace yannq {

template<typename T>
class Mean : public AbstractLayer<T> 
{
public:
	using Scalar = T;
	using Vector = typename AbstractLayer<T>::Vector;
	using Matrix = typename AbstractLayer<T>::Matrix;
	using VectorRef = typename AbstractLayer<T>::VectorRef;
	using VectorConstRef = typename AbstractLayer<T>::VectorConstRef;

	std::array<uint32_t, 2> inputDim_;
	uint32_t meanAxis_;

public:
	/// Constructor
	Mean(const std::array<uint32_t, 2>& inputDim, uint32_t meanAxis)
		: inputDim_(inputDim), meanAxis_{meanAxis}
	{
		assert((meanAxis == 0) || (meanAxis == 1));
	}

	Mean(const Mean& ) = default;
	Mean(Mean&& ) = default;

	Mean& operator=(const Mean&) = default;
	Mean& operator=(Mean&&) = default;

	bool operator==(const Mean& rhs) const
	{
		return (inputDim_ == rhs.inputDim_) && (meanAxis_ == rhs.meanAxis_);
	}
	
	bool operator!=(const Mean& rhs) const
	{
		return !(*this == rhs);
	}


	std::string name() const override { return "Mean layer"; }

	nlohmann::json desc() const override 
	{
		nlohmann::json res;
		res["name"] = name();
		res["input_dim"] = inputDim_;
		res["mean_axis"] = meanAxis_;
		return res;
	}

	template<class RandomEngine>
	void randomizeParams(RandomEngine&& re, remove_complex_t<Scalar> sigma)
	{
	}

	uint32_t paramDim() const override { return 0; }

	uint32_t inputDim() const 
	{
		return inputDim_[0]*inputDim_[1];
	}

	uint32_t outputDim() const
	{
		return inputDim_[1-meanAxis_];
	}

	uint32_t outputDim(uint32_t dim) const override {
		(void)dim;
		assert(inputDim() == dim);
		return outputDim();
	}

	Vector getParams() const override 
	{
		return Vector{};
	}

	void setParams(VectorConstRef /*pars*/) override 
	{
	}

	void updateParams(VectorConstRef /*ups*/) override
	{
	}

	// Feedforward
	void forward(const VectorConstRef& input, VectorRef output) override 
	{
		auto m = Eigen::Map<const Matrix>(input.data(), inputDim_[0], inputDim_[1]);
		if(meanAxis_ == 0)
		{
			output = m.colwise().sum()/m.rows();
		}
		else if(meanAxis_ == 1)
		{
			output = m.rowwise().sum()/m.cols();
		}
	}

	// Computes derivative.
	void backprop(const VectorConstRef& /*prev_layer_output*/,
			const VectorConstRef& /*this_layer_output*/,
			const VectorConstRef& dout,
			VectorRef din, VectorRef /*der*/) override 
	{
		// dout = d(L) / d(z)
		// Derivative for bias, d(L) / d(b) = d(L) / d(z)
		//
		// Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
		auto m = Matrix(inputDim_[0], inputDim_[1]);
		if(meanAxis_ == 0)
		{
			m.rowwise() = dout.transpose() / inputDim_[0];
		}
		else if(meanAxis_ == 1)
		{
			m.colwise() = dout / inputDim_[1];
		}
		din = Eigen::Map<const Vector>(m.data(), inputDim_[0]*inputDim_[1]);
	}

	uint32_t fanIn() override
	{
		return inputDim();
	}
	uint32_t fanOut() override
	{
		return outputDim();
	}
};

}  // namespace yannq

