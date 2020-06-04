#pragma once

#include<vector>
#include "Layer.h"

class LossLayer :public Layer
{
protected:
	float loss;
public:
	LossLayer():loss(0.f){ setLayerType(LayerType::Loss); }
};
class CrossEntropyLossLayer :public LossLayer
{
private:
	TensorXF diff;
public:
	CrossEntropyLossLayer(unsigned int numBatch, unsigned int numClass) :
		LossLayer(),
		diff(U{ numBatch, numClass }, 0.f)
		{}
	void forward(TensorXF& predict, Tensor<unsigned int>& label);
	TensorXF backward(TensorXF& predict, Tensor<unsigned int>& label);
};