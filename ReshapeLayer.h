#pragma once

#include<vector>
#include "Layer.h"

class ReshapeLayer :public Layer
{
public:
	ReshapeLayer(){ setLayerType(LayerType::Reshape); }
	void forward(const TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};