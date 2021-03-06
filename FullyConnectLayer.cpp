#include <vector>
#include <random>
#include <iostream>
#include "FullyConnectLayer.h"

void FullyConnectLayer::init()
{
	std::size_t weightSize = weight.size();
	std::size_t biaSize = bias.size();
	std::default_random_engine e;
	std::normal_distribution<float> n(0.,(float)sqrt(2./biaSize));
	for (std::size_t i = 0; i < weightSize; ++i)
	{
		weight[i] = n(e);
	}
	for (std::size_t i = 0; i < biaSize; ++i)
	{
		bias[i] = n(e);
	}
}

TensorXF FullyConnectLayer::forward(TensorXF& input)
{
	
	std::vector<unsigned int> inDim = input.dim();//b  c
	std::vector<unsigned int> biasDim = bias.dim();//nc
	TensorXF output(U{ inDim[0], biasDim[0] },0.f);
	std::vector<unsigned int> outDim = output.dim(); // b  nc
	preInput = input;
	for (unsigned int nb = 0; nb < outDim[0]; ++nb)
	{
		for (unsigned int nc = 0; nc < outDim[1]; ++nc)
		{
			output(U{ nb, nc }) = bias(U{ nc });
			for (unsigned int c = 0; c < inDim[1]; ++c)
			{
				output(U{ nb, nc }) += input(U{ nb, c })*weight(U{ c, nc });
			}
	//		std::cout << output(U{ nb, nc }) << "  ";
		}
	}
	return output;
}

TensorXF FullyConnectLayer::backward(TensorXF & input)
{
	// Y = WX + b
	// dx
	std::vector<unsigned int> preInputDim = preInput.dim();
	std::vector<unsigned int> weightDim = weight.dim();
	for (unsigned int nb = 0; nb < preInputDim[0]; ++nb)
	{
		for (unsigned int ni = 0; ni < preInputDim[1]; ++ni)
		{
			for (unsigned int no = 0; no < weightDim[1]; ++no)
			{
				dx(U{ nb, ni }) += weight(U{ ni, no }) * input(U{ nb, no });
			}
		}
	}
	//dw
	
	for (unsigned int ni = 0; ni < preInputDim[1]; ++ni)
	{
		for (unsigned int no = 0; no < weightDim[1]; ++no)
		{
			dw(U{ ni, no }) = 0.;
			for (unsigned int nb = 0; nb < preInputDim[0]; ++nb)
			{
				dw(U{ ni, no }) += preInput(U{ nb, ni })*input(U{ nb, no });
			}
		}
	}
	//db
	
	for (unsigned int no = 0; no < weightDim[1]; ++no)
	{
		db(U{ no }) = 0.;
		for (unsigned int nb = 0; nb < preInputDim[0]; ++nb)
		{
			db(U{ no }) += input(U{ nb, no });
		}
		
	}
	return TensorXF(dx);
}

void FullyConnectLayer::update()
{
	std::vector<unsigned int> weightDim = weight.dim();
	std::vector<unsigned int> dxDim = dx.dim();
	float b = 1.f / dxDim[0];
	db *= b;
	dw *= b;
	for (unsigned int nc = 0; nc < weightDim[1]; nc++)
	{
		bias(U{ nc }) -= lr*db(U{ nc });
	    for (unsigned int c = 0; c < weightDim[0];c++)
	
	    {
	    	weight(U{ c, nc }) -= lr * dw(U{ c, nc });
	    }
	}
}