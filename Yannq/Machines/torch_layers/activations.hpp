#ifndef YANNQ_MACHINES_TORCHLAYERS_ACTIVATIONS_HPP
#define YANNQ_MACHINES_TORCHLAYERS_ACTIVATIONS_HPP
#include <torch/torch.h>

/// a custom activation function
struct LeakyHardTanh : torch::nn::Module
{
	const double alpha_;
	LeakyHardTanh(const double alpha)
		: alpha_{alpha}
	{
	}
	torch::Tensor forward(const torch::Tensor& x)
	{
		return alpha_*x + (1-alpha_)*torch::nn::functional::hardtanh(x);
	}
};
struct LeakySoftShirink : torch::nn::Module
{
	const double alpha_;
	LeakySoftShirink(const double alpha)
		: alpha_{alpha}
	{
	}
	torch::Tensor forward(const torch::Tensor& x)
	{
		return alpha_*x + (1-alpha_)*torch::nn::functional::softshrink(x, 1.0);
	}
};

#endif//YANNQ_MACHINES_TORCHLAYERS_ACTIVATIONS_HPP
