#include <vector>
#include <random>
#include <iostream>
#include "ConvolutionLayer.h"

void ConvolutionLayer::init()
{
	std::default_random_engine e; 
	std::normal_distribution<float> n(0., 1.); 
	std::size_t weightSize = weight.size();
	for (std::size_t i = 0; i < weightSize; ++i)
	{
		weight[i] = n(e);
	}
	bias.fillData(0.f);
}

void ConvolutionLayer::forward( TensorXF& input, TensorXF& output)
{
	std::vector<unsigned int> inDim = input.dim();//b h w c
	std::vector<unsigned int> outDim = output.dim(); // b h w c
	std::vector<unsigned int> kernelDim = weight.dim();//h w ic oc
	int center_h = int(kernelDim[0] / 2), center_w = int(kernelDim[1] / 2);
	
	for (unsigned int nb = 0; nb < outDim[0]; nb++)
	{
		for (unsigned int nh = 0; nh < outDim[1]; nh++)
		{
			for (unsigned int nw = 0; nw < outDim[2]; nw++)
			{
				for (unsigned int nc = 0; nc < outDim[3]; nc++)
				{
					output(std::vector<unsigned int>{ nb, nh, nw, nc }) = bias(std::vector<unsigned int>{ nc });
					for ( unsigned int i = 0 ; 
						i<kernelDim[0]; i++)
					for ( unsigned int j = 0;
						j <kernelDim[1]; j++)
					if (nh + i<center_h ||  //nh-center_h+i<0
						nh + i >= inDim[1] + center_h ||
						nw + j < center_w ||
						nw + j >= inDim[2] + center_w){}
					else{
						for (unsigned int c = 0; c < inDim[3]; c++)
							output(std::vector<unsigned int>{ nb, nh, nw, nc }) +=
							input(std::vector<unsigned int>{ nb, nh + i - center_h, nw + j - center_w, c})
							* weight(std::vector<unsigned int>{ i, j,c,nc });
					}
						
				}
			}
		}
	}
	

}