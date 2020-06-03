#include <vector>
#include <random>
#include <iostream>
#include "FullyConnectLayer.h"

void FullyConnectLayer::init()
{
	std::default_random_engine e;
	std::normal_distribution<float> n(0., 1.);
	std::size_t weightSize = weight.size();
	std::size_t biaSize = bias.size();
	for (std::size_t i = 0; i < weightSize; ++i)
	{
		weight[i] = n(e);
	}
	for (std::size_t i = 0; i < biaSize; ++i)
	{
		bias[i] = n(e);
	}
}

void FullyConnectLayer::forward(TensorXF& input, TensorXF& output)
{
	std::vector<unsigned int> inDim = input.dim();//b  c
	std::vector<unsigned int> outDim = output.dim(); // b  nc
	for (unsigned int nb = 0; nb < outDim[0]; ++nb)
	{
		for (unsigned int nc = 0; nc < outDim[1]; ++nc)
		{
			output(U{ nb, nc }) = bias(U{ nc });
			for (unsigned int c = 0; c < inDim[1]; ++c)
			{
				output(U{ nb, nc }) += input(U{ nb, c })*weight(U{ c, nc });
			}
		}
	}
}