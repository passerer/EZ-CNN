#include <algorithm>
#include <vector>
#include "ActivationLayer.h"
#include "MathFunctions.h"

TensorXF SigmoidLayer::forward( TensorXF& input)
{
	std::vector<unsigned int> inDim = input.dim();
	TensorXF output(inDim, 0.f);
	std::size_t size = input.size();
	std::size_t offset = 0;
	for (; offset < size; ++offset)
	{
		output[offset] = sigmoid(input[offset]);
	}
	y = output;
	return output;
	
}
TensorXF SigmoidLayer::backward( TensorXF& input)
{
	std::vector<unsigned int> inDim = input.dim();
	TensorXF output(inDim, 0.f);
	std::size_t size = input.size();
	std::size_t offset = 0;
	for (; offset < size; ++offset)
	{
		output[offset] =  input[offset]*y[offset]*(1-y[offset]);
	}
	return output;
}

TensorXF TanhLayer::forward(TensorXF& input)
{
	std::vector<unsigned int> inDim = input.dim();
	TensorXF output(inDim, 0.f);
	std::size_t size = input.size();
	std::size_t offset = 0;
	for (; offset < size; ++offset)
	{
		output[offset] = tanh_(input[offset]);
	}
	return output;

}
/*
TensorXF TanhLayer::backward(TensorXF& input)
{

}
*/
TensorXF ReluLayer::forward( TensorXF& input)
{
	std::vector<unsigned int> inDim = input.dim();
	TensorXF output(inDim, 0.f);
	std::size_t size = input.size();
	std::size_t offset = 0;
	for (; offset < size; ++offset)
	{
		if (input[offset] >0.f){ pos[offset] = 1.f; }
		output[offset] = relu(input[offset]);
	}
	return output;

}
TensorXF ReluLayer::backward(TensorXF& input)
{
	std::vector<unsigned int> inDim = input.dim();
	TensorXF output(inDim, 0.f);
	std::size_t size = input.size();
	std::size_t offset = 0;
	for (; offset < size; ++offset)
	{
		output[offset] = pos[offset] * input[offset];
	}
	return output;
}



