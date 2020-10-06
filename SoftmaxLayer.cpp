#include <vector>
#include <cmath>
#include <iostream>
#include "SoftmaxLayer.h"

TensorXF SoftmaxLayer::forward( TensorXF& input)
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
			output(U{ nb, nc }) = std::exp(input(U{ nb, nc })) / (sum + 0.001f);
		}
	}
	return TensorXF (output);
}


TensorXF SoftmaxLayer::backward( TensorXF& input)
{
	std::vector<unsigned int> inDim = input.dim();
	std::vector<unsigned int> diffDim = diff.dim();//b  c
	for (unsigned int nb = 0; nb < diffDim[0]; ++nb)
	{
		for (unsigned int nc = 0; nc < diffDim[1]; ++nc)
		{
			diff(U{ nb, nc }) = 0.f;
			for (unsigned int c = 0; c < inDim[1]; ++c)
			{
				if (nc != c)
				{
					diff(U{ nb, nc }) -= input(U{ nb, c })*output(U{ nb, c })*output(U{ nb, nc });
				}
				else
				{
					diff(U{ nb, nc }) += input(U{ nb, c })*output(U{ nb, c })*(1 - output(U{ nb, nc }));
				}
			}
		}
	}
	return TensorXF(diff);
}

