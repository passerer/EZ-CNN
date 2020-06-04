#pragma once

#include<vector>
#include "Tensor.h"

enum class Phase
{
	Train,
	Test
};
enum class LayerType
{
	Activation,
	BatchNormalization,
	Convolution,
	FullConnect,
	Dropout,
	Input,
	Pooling,
	Softmax,
	Reshape,
	Loss
};

class Layer{

protected :
	Phase phase = Phase::Train;
	LayerType layerType;
	float lr = 0.1f;

public:
	Layer(){}
	~Layer(){}
	 void forward(const TensorXF& input, TensorXF& output);
	 void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff) ;

	LayerType getLayerType(LayerType _layerType){ return layerType; }
	void setLayerType(LayerType _layerType){  layerType = _layerType; }
	inline void setLR(const float value){ lr = value; }
	inline float getLR(void) const { return lr; }
	inline void setPhase(Phase _phase) { phase = _phase; }
	inline Phase getPhase(void){ return phase; }
};