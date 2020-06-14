#include <vector>
#include"ReshapeLayer.h"

TensorXF ReshapeLayer::forward(TensorXF& input)
{
	std::vector<unsigned int> inDim = input.dim();
	for (unsigned int nb = 0; nb < inDim[0]; nb++)
	{
		for (unsigned int nh = 0; nh < inDim[1]; nh++)
		{
			for (unsigned int nw = 0; nw < inDim[2]; nw++)
			{
				for (unsigned int nc = 0; nc < inDim[3]; nc++)
				{
					x(U{ nb, nh*inDim[2]*inDim[3] + nw*inDim[3] + nc })
						= input(U{ nb, nh, nw, nc });
				}
			}
		}
	}
	return TensorXF(x);
}

TensorXF ReshapeLayer::backward(TensorXF& input)
{
	std::vector<unsigned int> dxDim = dx.dim();
	for (unsigned int nb = 0; nb < dxDim[0]; nb++)
	{
		for (unsigned int nh = 0; nh < dxDim[1]; nh++)
		{
			for (unsigned int nw = 0; nw < dxDim[2]; nw++)
			{
				for (unsigned int nc = 0; nc < dxDim[3]; nc++)
				{
					dx(U{ nb, nh, nw, nc })
						= input(U{ nb, nh*dxDim[2] * dxDim[3] + nw*dxDim[3] + nc });
				}
			}
		}
	}
	return TensorXF(dx);
}