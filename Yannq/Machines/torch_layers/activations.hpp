#ifndef YANNQ_MACHINES_TORCHLAYERS_ACTIVATIONS_HPP
#define YANNQ_MACHINES_TORCHLAYERS_ACTIVATIONS_HPP
#include <torch/torch.h>

namespace F = torch::nn::functional;

/// a custom activation function
torch::Tensor leakyHardTanh(torch::Tensor input, const double alpha)
{
	return alpha*input + (1-alpha)*F::hardtanh(input);
}
torch::Tensor leakySoftShirink(torch::Tensor input, const double alpha)
{
	return alpha*input + (1-alpha)*softshrink(input, 1.0);
}
#endif//YANNQ_MACHINES_TORCHLAYERS_ACTIVATIONS_HPP
