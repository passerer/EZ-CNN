#pragma once

#include<vector>
#include "Layer.h"

class SoftmaxLayer :public Layer
{
public:
	SoftmaxLayer(){ setLayerType(LayerType::Softmax); }
	void forward( TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};