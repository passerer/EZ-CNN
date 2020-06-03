#include <vector>
#include <iostream>
#include <limits>
#include "PoolingLayer.h"

void MaxPoolingLayer::forward(TensorXF& input, TensorXF& output)
{
	std::vector<unsigned int> inDim = input.dim();//b h w  c
	std::vector<unsigned int> outDim = output.dim(); // b h/s,w/s c
	for (unsigned int nb = 0; nb < outDim[0]; ++nb)
	{
		for (unsigned int nh = 0; nh < outDim[1]; ++nh)
		{
			for (unsigned int nw = 0; nw < outDim[2]; ++nw)
			{
				for (unsigned int nc = 0; nc < outDim[3]; ++nc)
				{
					unsigned int h = 2 * nh, w = 2 * nw;
					unsigned int maxh = h, maxw = w;
					float _max = std::numeric_limits<float>::min();
					for (unsigned int i = 0; i < stride;++i)
					for (unsigned int j = 0; j < stride; ++j)
					{
						if (h + i < inDim[1]&& w + j<inDim[2]&& input(U{ nb, h+i, w+j, nc })>_max)
						{
							maxh = h + i; maxw = w + j;
							_max = input(U{ nb, h, w, nc });
						}
					}
					output(U{ nb, nh, nw, nc }) = input(U{ nb, maxh, maxw, nc });
				}
			}
		}
	}
}

void AveragePoolingLayer::forward(TensorXF& input, TensorXF& output)
{
	std::vector<unsigned int> inDim = input.dim();//b h w  c
	std::vector<unsigned int> outDim = output.dim(); // b h/s,w/s c
	for (unsigned int nb = 0; nb < outDim[0]; ++nb)
	{
		for (unsigned int nh = 0; nh < outDim[1]; ++nh)
		{
			for (unsigned int nw = 0; nw < outDim[2]; ++nw)
			{
				for (unsigned int nc = 0; nc < outDim[3]; ++nc)
				{
					unsigned int h = 2 * nh, w = 2 * nw,count =0;
					float sum = 0.f;
					float _max = std::numeric_limits<float>::min();
					for (unsigned int i = 0; i < stride; ++i)
					for (unsigned int j = 0; j < stride; ++j)
					{
						if (h + i < inDim[1] && w + j<inDim[2])
						{
							sum += input(U{ nb, h + i, w + j, nc });
							count++;
						}
					}
					output(U{ nb, nh, nw, nc }) = sum / count;
				}
			}
		}
	}
}