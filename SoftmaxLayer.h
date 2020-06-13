#pragma once

#include<vector>
#include "Layer.h"

class SoftmaxLayer :public Layer
{
private:
	TensorXF diff;
	TensorXF output;
public:
	SoftmaxLayer(unsigned int nb, unsigned int nc) :diff(U{ nb, nc }, 0.f), output(U{ nb, nc }, 0.f)
	{ setLayerType(LayerType::Softmax); }
	TensorXF forward( TensorXF& input);
	TensorXF backward(TensorXF& input);
	TensorXF& getoutput();
	void update();
};