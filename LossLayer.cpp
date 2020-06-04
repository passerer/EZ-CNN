
#include<cmath>
#include "LossLayer.h"

void CrossEntropyLossLayer::forward(TensorXF& predict, Tensor<unsigned int>& label)
{
	std::vector<unsigned int> inDim = predict.dim();
	for (unsigned int nb = 0; nb < inDim[0]; ++nb)
	{
		unsigned int index = label(U{ nb });
		loss -= std::log(predict(U{ nb, index }));
	}
	loss /= inDim[0];
}