#ifndef YANNQ_MACHINES_CPPTORCH_CONV1D_HPP
#define YANNQ_MACHINES_CPPTORCH_CONV1D_HPP
#include <torch/torch.h>

struct Conv1D : torch::nn::Module
{
	bool periodic_;

	int inChannels_;
	int outChannels_;
	int kernel_size_;

	torch::Tensor wight_;
	torch::Tensor bias_;

	Conv1D(bool periodic, int inChannels, int outChannels, int kernel_size, bool useBias = false)
		: periodic_{periodic}, inChannels_{inChannels}, outChannels_{outChannels}
		kernel_size_{kernel_size}, periodic_{periodic}
	{
		weight_ = register_parameter("weight", torch::randn{outChannels_, inChannels_, kernel_size});
		if(useBias)
			bias_ = register_parameter("bias", torch::randn{outChannels_});
		else
			bias_ = {};
	}

	torch::Tensor forward(torch::Tensor x)
	{
		constexpr auto dim1 = -1;

		if(periodic_)
		{
			x = torch::cat({x.slice(dim1, -1), x, x.slice(dim1, 0, 1)}, dim1);
		}
		return torch::conv1d(x, weight_, bias_);
	}
};
}// namespace yannq
#endif//YANNQ_MACHINES_CPPTORCH_CONV1D_HPP
