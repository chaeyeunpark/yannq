#include <torch/torch.h>
#include <Machines/torch_layers/Conv1D.cpp>
#include <Machines/torch_layers/activations.hpp>
#include <Machines/AmplitudePhaseTorch.cpp>

#include <iostream>
#include <GroundState/SRMatExact.hpp>
#include <Hamiltonians/XXXJ1J2.hpp>
#include <Basis/BasisJz.hpp>

#include <Optimizers/SGD.hpp>

#include <Utilities/Utility.hpp>

namespace F = torch::nn::functional;

using namespace yannq;
using namespace Eigen;

struct Net : torch::nn::Module
{
	std::shared_ptr<Conv1D> conv1;
	std::shared_ptr<Conv1D> conv2;
	std::shared_ptr<Conv1D> conv3;
	std::shared_ptr<Conv1D> conv4;
	std::shared_ptr<Conv1D> conv5;
	std::shared_ptr<Conv1D> conv6;
	torch::nn::Linear fc{nullptr};

	int inputSize_;
	const int kernel_size = 3;
	const double alpha = 0.3;

	Net(int inputSize)
		: inputSize_{inputSize}
	{
		conv1 = register_module("conv1", std::make_shared<Conv1D>(true, 1, 12, kernel_size, false));
		conv2 = register_module("conv2", std::make_shared<Conv1D>(true, 12, 10, kernel_size, false));
		conv3 = register_module("conv3", std::make_shared<Conv1D>(true, 10, 8, kernel_size, false));
		conv4 = register_module("conv4", std::make_shared<Conv1D>(true, 8, 6, kernel_size, false));
		conv5 = register_module("conv5", std::make_shared<Conv1D>(true, 6, 4, kernel_size, false));
		conv6 = register_module("conv6", std::make_shared<Conv1D>(true, 4, 2, kernel_size, false));

		fc = register_module("fc", torch::nn::Linear(2*inputSize, 1));
	}

	torch::Tensor forward(torch::Tensor x) {
		// Use one of many tensor manipulation functions.
		x = conv1->forward(x);
		x = leakyHardTanh(x, alpha);
		x = conv2->forward(x);
		x = leakyHardTanh(x, alpha);
		x = conv3->forward(x);
		x = leakyHardTanh(x, alpha);
		x = conv4->forward(x);
		x = leakyHardTanh(x, alpha);
		x = conv5->forward(x);
		x = leakyHardTanh(x, alpha);
		x = conv6->forward(x);
		x = leakyHardTanh(x, alpha);
		x = fc->forward(x.reshape({x.size(0), 2*inputSize_}));
		x = F::softsign(x);
		return x;
	}
};


MatrixXd jacobian_auto(Net& net, torch::Tensor input)
{
	const int nSmp = input.size(0);
	std::vector<int> paramSizes;
	auto params = net.parameters();
	for(const auto p : params)
	{
		paramSizes.emplace_back(p.numel());
	}

	const int paramSize = std::accumulate(paramSizes.begin(), paramSizes.end(), 0);

	MatrixXd jacobian(nSmp, paramSize);

	auto out = net.forward(input);
	
	torch::Tensor z = torch::zeros({nSmp,1}, torch::dtype(torch::kFloat64));
	auto z_a = z.accessor<double,2>();
	for(int n = 0; n < nSmp; ++n)
	{
		z_a[n][0] = 1.0;
		net.zero_grad();
		out.backward(z, true);

		int s = 0;
		for(const auto p : params)
		{
			torch::Tensor g = torch::flatten(p.grad());
			g = g.contiguous();
			jacobian.block(n, s, 1, g.numel()) = 
				Eigen::Map<const RowVectorXd>(g.data_ptr<double>(), 1, g.numel());
			s += g.numel();
		}
		z_a[n][0] = 0.0;
	}
	return jacobian;
}

MatrixXd jacobian_nume(Net& net, torch::Tensor input)
{
	const double h = 1e-5;
	const int nSmp = input.size(0);
	std::vector<int> paramSizes;
	auto params = net.parameters();
	for(const auto p : params)
	{
		paramSizes.emplace_back(p.numel());
	}
	const int paramSize = std::accumulate(paramSizes.begin(), paramSizes.end(), 0);

	MatrixXd jacobian(nSmp, paramSize);

	net.zero_grad();
	
	int s = 0;
	for(auto p: params)
	{
		p = p.detach();
		torch::Tensor o = torch::zeros({p.numel()}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false));
		auto o_a = o.accessor<double, 1>();
		for(int n = 0; n < p.numel(); n++)
		{
			o_a[n] = h;
			p.add_(o.reshape_as(p));
			auto out1 = net.forward(input).clone().detach();
			p.add_( -2*o.reshape_as(p) );
			auto out2 = net.forward(input).clone().detach();
			o_a[n] = 0;

			torch::Tensor der = (out1 - out2)/(2*h);

			jacobian.block(0, s+n, nSmp, 1) = Eigen::Map<const VectorXd>(der.data_ptr<double>(), nSmp);
		}

		s += p.numel();
	}

	return jacobian;
}

int main()
{
	const int N = 12;
	const int nSmp = 6;

	std::random_device rd;
	std::default_random_engine re{rd()};
	
	/*
	auto phaseNet = Net{N};
	phaseNet.to(torch::kFloat64);
	auto options = torch::TensorOptions().dtype(torch::kFloat64);
	torch::Tensor x = torch::randn({nSmp,1,N}, options);
	auto jacobian_1 = jacobian_auto(phaseNet, x);
	auto jacobian_2 = jacobian_nume(phaseNet, x);

	std::cout << (jacobian_1 - jacobian_2).norm() << std::endl;
	*/
	auto machine = AmplitudePhase<Net>(N, 3*N, std::make_shared<Net>(N));
	machine.initializeAmplitudeRandom(re, 1e-3);

	Eigen::MatrixXi sigmas = randomSigma(N*nSmp, re);
	sigmas.resize(N, nSmp);

	auto res = machine.logDeriv(sigmas);
	std::cout << std::get<0>(res) << std::endl;
	std::cout << std::get<1>(res) << std::endl;
	std::cout << std::get<2>(res) << std::endl;
	return 0;
}
