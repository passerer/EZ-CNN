#pragma once

#include<vector>
#include "Layer.h"

class DropOutLayer :public Layer
{
public:
	DropOutLayer(){ setLayerType(LayerType::Dropout); }
	void forward(const TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};