
#include<cmath>
#include "LossLayer.h"

void CrossEntropyLossLayer::forward(TensorXF& predict, Tensor<unsigned int>& label)
{
	std::vector<unsigned int> inDim = predict.dim();
	for (unsigned int nb = 0; nb < inDim[0]; ++nb)
	{
		unsigned int index = label(U{ nb });
		loss -= std::log(predict(U{ nb, index })+1e-8);
	}
	loss /= inDim[0];
}

TensorXF CrossEntropyLossLayer::backward(TensorXF& predict, Tensor<unsigned int>& label)
{
	std::vector<unsigned int> inDim = predict.dim();
	for (unsigned int nb = 0; nb < inDim[0]; ++nb){
		for (unsigned int nc = 0; nc < inDim[1]; ++nc )
		{
			diff(U{ nb, nc }) = 0;
		}
	}
	for (unsigned int nb = 0; nb < inDim[0]; ++nb)
	{
		unsigned int index = label(U{ nb });
		diff(U{ nb, index }) = -1.f / (predict(U{ nb, index }) + 1e-8);
	}
	TensorXF out(diff);
	return out;
}