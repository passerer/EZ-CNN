#include <vector>
#include <cmath>
#include <iostream>
#include "SoftmaxLayer.h"

void SoftmaxLayer::forward( TensorXF& input, TensorXF& output)
{
	std::vector<unsigned int> inDim = input.dim();//b  c
	std::vector<unsigned int> outDim = output.dim(); // b c
	for (unsigned int nb = 0; nb < outDim[0]; ++nb)
	{
		float sum = 0.f;
		for (unsigned int nc = 0; nc < inDim[1]; ++nc)
		{
			sum += std::exp(input(U{ nb, nc }));
		}
		for (unsigned int nc = 0; nc < outDim[1]; ++nc)
		{
			output(U{ nb, nc }) = std::exp(input(U{ nb, nc })) / sum;
		}
	}
}


void SoftmaxLayer::backward(const TensorXF& input, const TensorXF& output,
	const TensorXF& preDiff, TensorXF& nextDiff)
{
}