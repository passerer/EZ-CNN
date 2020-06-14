#pragma once 
#include"Layer.h"

class ActivationLayer :public Layer
{

public:
	ActivationLayer() 
	{ setLayerType(LayerType::Activation); }
};
class SigmoidLayer : public ActivationLayer
{
private:
	TensorXF y;
public:
	SigmoidLayer(U dim) :y(dim, 0.f){}

	SigmoidLayer(unsigned int nb, unsigned int nh, unsigned int nw, unsigned int nc)
		:y(U{ nb, nh, nw, nc }, 0.f){}

	SigmoidLayer(unsigned int nb, unsigned int nc)
		:y(U{ nb, nc }, 0.f){ }
	 TensorXF forward( TensorXF& input) ;
	 TensorXF backward( TensorXF& input) ;
};

class TanhLayer : public ActivationLayer
{
public:
//	TanhLayer();
//	virtual ~TanhLayer();

	TensorXF forward(TensorXF& input);
	TensorXF backward(TensorXF& input);
};

class ReluLayer : public ActivationLayer
{
private:
	TensorXF pos;
	TensorXF dx;
public:
	ReluLayer(U dim) :pos(dim, 0.f), dx(dim, 0.f)
	{
	}

	ReluLayer(unsigned int nb, unsigned int nh, unsigned int nw, unsigned int nc)
		:pos(U{ nb, nh, nw, nc }, 0.f), dx(U{ nb, nh, nw, nc }, 0.f)
	{
	}

	ReluLayer(unsigned int nb, unsigned int nc)
		:pos(U{ nb, nc }, 0.f), dx(U{ nb, nc }, 0.f)
	{
	}

	TensorXF forward(TensorXF& input);
	TensorXF backward(TensorXF& input);
};