#ifndef YANNQ_MACHINES_CPPTORCH_CONV2D_HPP
#define YANNQ_MACHINES_CPPTORCH_CONV2D_HPP
#include <torch/torch.h>

struct Conv2D : torch::nn::Module
{
	bool periodic_;

	int inChannels_;
	int outChannels_;
	int kernel_size_;

	torch::Tensor wight_;
	torch::Tensor bias_;

	Conv2D(bool periodic, int inChannels, int outChannels, int kernel_size, bool useBias = false)
		: periodic_{periodic}, inChannels_{inChannels}, outChannels_{outChannels}
		kernel_size_{kernel_size}, periodic_{periodic}
	{
		weight_ = register_parameter("weight", torch::randn{outChannels_, inChannels_, kernel_size, kernel_size});
		if(useBias)
			bias_ = register_parameter("bias", torch::randn{outChannels_});
		else
			bias_ = {};
	}

	torch::Tensor forward(torch::Tensor x)
	{
		constexpr auto dim1 = -1;
		constexpr auto dim2 = -2;

		if(periodic_)
		{
			x = torch::cat({x.slice(dim1, -1), x, x.slice(dim1, 0, 1)}, dim1);
			x = torch::cat({x.slice(dim2, -1), x, x.slice(dim2, 0, 1)}, dim2);
		}
		return torch::conv2d(x, weight_, bias);
	}
};
}// namespace yannq
#endif//YANNQ_MACHINES_CPPTORCH_CONV2D_HPP
