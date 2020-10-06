
#include<cmath>
#include "LossLayer.h"

float CrossEntropyLossLayer::forward(TensorXF& predict, Tensor<unsigned int>& label)
{
	std::vector<unsigned int> inDim = predict.dim();
	for (unsigned int nb = 0; nb < inDim[0]; ++nb)
	{
		unsigned int index = label(U{ nb });
		loss -= std::log(predict(U{ nb, index }) + 0.0001f);
	}
	loss /= inDim[0];
	return loss;
}

TensorXF CrossEntropyLossLayer::backward(TensorXF& predict, Tensor<unsigned int>& label)
{
	std::vector<unsigned int> inDim = predict.dim();
	for (unsigned int nb = 0; nb < inDim[0]; ++nb){
		for (unsigned int nc = 0; nc < inDim[1]; ++nc)
		{
			diff(U{ nb, nc }) = 0;
		}
	}
	for (unsigned int nb = 0; nb < inDim[0]; ++nb)
	{
		unsigned int index = label(U{ nb });
		diff(U{ nb, index }) = -1.f / (predict(U{ nb, index }) + 0.0001f);
	}
	TensorXF out(diff);
	return out;
}

float MeanSquareError2DLayer::forward(TensorXF& predict, Tensor<unsigned int>& label)
{
	loss = 0.;
	std::vector<unsigned int> inDim = predict.dim();
	for (unsigned int nb = 0; nb < inDim[0]; ++nb)
	{
		unsigned int index = label(U{ nb });
		for (unsigned int nc = 0; nc < inDim[1]; ++nc)
		{
			if (nc == index) loss += (predict(U{ nb, nc })-1)*(predict(U{ nb, nc }) - 1);
			else loss += predict(U{ nb, nc })*predict(U{ nb, nc });
		}
	}
	loss /= inDim[0]*inDim[1];
	return loss;
}

TensorXF MeanSquareError2DLayer::backward(TensorXF& predict, Tensor<unsigned int>& label)
{
	std::vector<unsigned int> inDim = predict.dim();
	for(unsigned int nb = 0; nb < inDim[0]; ++nb)
	{
		unsigned int index = label(U{ nb });
		for (unsigned int nc = 0; nc < inDim[1]; ++nc)
		{
			if (nc == index) diff(U{ nb, nc }) = 2*(predict(U{ nb, nc }) - 1);
			else  diff(U{ nb, nc }) = predict(U{ nb, nc });
		}
	}
	TensorXF out(diff);
	return out;
}