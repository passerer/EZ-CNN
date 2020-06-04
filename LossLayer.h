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
public:
	void forward(TensorXF& predict, Tensor<unsigned int>& label);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};