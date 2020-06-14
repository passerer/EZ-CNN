#include <vector>
#include <iostream>
#include <limits>
#include "PoolingLayer.h"

TensorXF MaxPoolingLayer::forward(TensorXF& input)
{
	std::vector<unsigned int> inDim = input.dim();//b h w  c
	std::vector<unsigned int> outDim = output.dim(); // b h/s,w/s c
	for (unsigned int nb = 0; nb < inDim[0]; ++nb)
	{
		for (unsigned int nh = 0; nh < inDim[1]; ++nh)
		{
			for (unsigned int nw = 0; nw < inDim[2]; ++nw)
			{
				for (unsigned int nc = 0; nc < inDim[3]; ++nc)
				{
					pos(U{ nb, nh, nw, nc }) = 0.f;
				}
			}
		}
	}

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
					pos(U{ nb, nh, nw, nc }) = 1.f;
					output(U{ nb, nh, nw, nc }) = input(U{ nb, maxh, maxw, nc });
				}
			}
		}
	}
	return TensorXF(output);
}

TensorXF MaxPoolingLayer::backward(TensorXF& input)
{
	std::vector<unsigned int> outDim = input.dim();//b h/s w/s  c
	outDim[1] *= stride, outDim[2] *= stride;
	TensorXF output(outDim, 0.f);
	for (unsigned int nb = 0; nb < outDim[0]; ++nb)
	{
		for (unsigned int nh = 0; nh < outDim[1]; ++nh)
		{
			for (unsigned int nw = 0; nw < outDim[2]; ++nw)
			{
				for (unsigned int nc = 0; nc < outDim[3]; ++nc)
				{
					output(U{ nb, nh, nw, nc }) = input(U{ nb, nh / stride, nw / stride, nc })*pos(U{ nb, nh, nw, nc });
				}
			}
		}
	}
	return output;
}

TensorXF AveragePoolingLayer::forward(TensorXF& input)
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

TensorXF AveragePoolingLayer::backward(TensorXF& input)
{
	std::vector<unsigned int> outDim = input.dim();//b h/s w/s  c
	outDim[1] *= stride, outDim[2] *= stride;
	TensorXF output(outDim, 0.f);
	for (unsigned int nb = 0; nb < outDim[0]; ++nb)
	{
		for (unsigned int nh = 0; nh < outDim[1]; ++nh)
		{
			for (unsigned int nw = 0; nw < outDim[2]; ++nw)
			{
				for (unsigned int nc = 0; nc < outDim[3]; ++nc)
				{
					output(U{ nb, nh, nw, nc }) = input(U{ nb, nh / stride, nw / stride, nc })/(stride*stride);
				}
			}
		}
	}
	return output;
}
