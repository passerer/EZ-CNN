#pragma once
#include <cmath>
#include <random>

static inline float sigmoid(const float x)
{
	float result = 0.f;
	result = 1.f / (1.f + std::exp(-1.f*x));
	return result;
}

static inline float tanh_(const float x)
{
	float result = 0.f;
	result = 2 * sigmoid(2 * x) - 1;
	return result;
}

static inline float relu(const float x)
{
	float result = 0.f;
	result = x > 0.f ? x : 0.f;
	return result;
}

