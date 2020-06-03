#pragma once 
#include"Layer.h"

class ActivationLayer :public Layer
{
public:
	ActivationLayer() { setLayerType(LayerType::Activation); }
};
class SigmodLayer : public ActivationLayer
{
public:
//	SigmodLayer();
//	virtual ~SigmodLayer();

	 void forward(const TensorXF& input, TensorXF& output) ;
	 void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff) ;
};

class TanhLayer : public ActivationLayer
{
public:
//	TanhLayer();
//	virtual ~TanhLayer();

    void forward(const TensorXF& input, TensorXF& output) ;
    void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff) ;
};

class ReluLayer : public ActivationLayer
{
public:
//	ReluLayer();
//	virtual ~ReluLayer();

	 void forward(const TensorXF& input, TensorXF& output) ;
	 void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff) ;
};