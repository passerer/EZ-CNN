#include <algorithm>
#include <vector>
#include "ActivationLayer.h"
#include "MathFunctions.h"

void SigmodLayer::forward(const TensorXF& input, TensorXF& output)
{
	std::size_t size = input.size();
	std::size_t offset = 0;
	for (; offset < size; ++offset)
	{
		output[offset] = sigmoid(input[offset]);
	}
	
}
void SigmodLayer::backward(const TensorXF& input, const TensorXF& output,
	const TensorXF& preDiff, TensorXF& nextDiff)
{

}

void TanhLayer::forward(const TensorXF& input, TensorXF& output)
{
	std::size_t size = input.size();
	std::size_t offset = 0;
	for (; offset < size; ++offset)
	{
		output[offset] = tanh_(input[offset]);
	}

}
void TanhLayer::backward(const TensorXF& input, const TensorXF& output,
	const TensorXF& preDiff, TensorXF& nextDiff)
{

}
void ReluLayer::forward(const TensorXF& input, TensorXF& output)
{
	std::size_t size = input.size();
	std::size_t offset = 0;
	for (; offset < size; ++offset)
	{
		output[offset] = relu(input[offset]);
	}

}
void ReluLayer::backward(const TensorXF& input, const TensorXF& output,
	const TensorXF& preDiff, TensorXF& nextDiff)
{

}



