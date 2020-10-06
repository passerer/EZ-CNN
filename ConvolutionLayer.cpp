#include <vector>
#include <random>
#include <iostream>
#include "ConvolutionLayer.h"

void ConvolutionLayer::init()
{
	std::vector<unsigned int> kernelDim = weight.dim();//h w ic oc
	unsigned int N = kernelDim[0] * kernelDim[1] * kernelDim[2];
	std::default_random_engine e; 
	std::normal_distribution<float> n(0.,(float)sqrt(2./N)); 
	std::size_t weightSize = weight.size();
	for (std::size_t i = 0; i < weightSize; ++i)
	{
		weight[i] = n(e) ;
	}
	bias.fillData(0.f);
}

TensorXF ConvolutionLayer::forward( TensorXF& input)
{
	preInput = input;
	std::vector<unsigned int> inDim = input.dim();//b h w c
	std::vector<unsigned int> outDim = inDim; // b h w c
	std::vector<unsigned int> kernelDim = weight.dim();//h w ic oc
	outDim[3] = kernelDim[3];
	TensorXF output(outDim, 0.f);
	unsigned int center_h = int(kernelDim[0] / 2), center_w = int(kernelDim[1] / 2);
	
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
			//		std::cout << output(std::vector<unsigned int>{ nb, nh, nw, nc }) << "  ";
				}
			}
		}
	}
	
	return output;
}

TensorXF ConvolutionLayer::backward(TensorXF& input)
{
	std::vector<unsigned int> dbDim = db.dim();//nc
	std::vector<unsigned int> inDim = input.dim();//b , h, w, nc
	std::vector<unsigned int> kernelDim = weight.dim();//h w ic oc
	// db
	for (unsigned int nc = 0; nc < dbDim[0]; nc++)
	{
		db(U{ nc }) = 0.f;
		for (unsigned int nb = 0; nb < inDim[0];nb++)
		for (unsigned int nh = 0; nh < inDim[1];nh++)
		for (unsigned int nw = 0; nw < inDim[2]; nw++)
		{
			db(U{ nc }) += input(U{ nb, nh, nw, nc });
		}
		db(U{ nc }) /= inDim[0];
	}
	//dw
	
	std::vector<unsigned int> dwDim = dw.dim();//k, l, c, nc
	unsigned int center_h = int(dwDim[0] / 2);
	unsigned int center_w = int(dwDim[1] / 2);
	for (unsigned int nk = 0; nk < dwDim[0]; nk++)
	{
		for (unsigned int nl = 0; nl < dwDim[1]; nl++)
		{
			for (unsigned int c = 0; c < dwDim[2]; c++)
			{
				for (unsigned int nc = 0; nc < dwDim[3]; nc++)
				{
					dw(U{ nk, nl, c, nc }) = 0.f;
					for (unsigned int nh = 0; nh < inDim[1]; nh++)
					{
						if (nh + nk < center_h || nh + nk >= center_h+inDim[1])
						{
							dw(U{ nk,nl,c,nc }) += 0.f;
						}
						else
						{
							unsigned int m_h = nh + nk-center_h;
							for (unsigned int nw = 0; nw < inDim[2]; nw++)
							{
								if (nw + nl < center_w || nw + nl >= inDim[2] + center_w)
								{
									dw(U{ nk, nl, c, nc }) += 0.f;
								}
								else
								{
									unsigned int m_w = nw + nl - center_w ;
									for (unsigned int nb = 0; nb < inDim[0]; nb++)
									{
										dw(U{ nk, nl, c, nc }) += input(U{ nb, m_h, m_w, nc })*preInput(U{ nb, m_h, m_w, c});
									}
									dw(U{ nk, nl, c, nc }) /= inDim[0];
								}
							}
						}
					}
				}
			}
		}
	}

	//dx
	std::vector<unsigned int> preInDim = preInput.dim();//b , h, w, c
	TensorXF output(preInDim, 0.f);
	for (unsigned int nb = 0; nb < preInDim[0]; nb++)
	{
		for (unsigned int nh = 0; nh < preInDim[1]; nh++)
		{
			for (unsigned int nw = 0; nw < preInDim[2]; nw++)
			{
				for (unsigned int c = 0; c < preInDim[3]; c++)
				{

					for (unsigned int i = 0;
						i<kernelDim[0]; i++)
					for (unsigned int j = 0;
						j <kernelDim[1]; j++)
					if (nh + i<center_h ||  //nh-center_h+i<0
						nh + i >= inDim[1] + center_h ||
						nw + j < center_w ||
						nw + j >= inDim[2] + center_w){
					}
					else{
						for (unsigned int nc = 0; nc < inDim[3]; nc++)
							output(U{ nb, nh, nw, c }) +=
							input(U{ nb, nh + i - center_h, nw + j - center_w, nc})
							* weight(U{ kernelDim[0] - 1 - i, kernelDim[1]-1-j, c, nc });
					}
				}
			}
		}
	}
	return output;
}

void ConvolutionLayer::update()
{
	std::vector<unsigned int> weightDim = weight.dim();
	std::vector<unsigned int> biasDim = bias.dim();
	for (unsigned int nk = 0; nk < weightDim[0]; nk++)
	{
		for (unsigned int nl = 0; nl < weightDim[1]; nl++)
		{
			for (unsigned int c = 0; c < weightDim[2]; c++)
			{
				for (unsigned int nc = 0; nc < weightDim[3]; nc++)
				{
					weight(U{ nk, nl, c, nc }) -= lr* dw(U{ nk, nl, c, nc });
				}
			}
		}
	}

	for (unsigned int nc = 0; nc < biasDim[0]; ++nc)
	{
		bias(U{ nc }) -= db(U{ nc });
	}
}