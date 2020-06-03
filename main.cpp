#include<iostream>
#include<vector>
#include <random>
#include"Tensor.h"
#include"ActivationLayer.h"
#include"ConvolutionLayer.h"
#include"PoolingLayer.h"

using namespace std;
int main()
{
	vector<unsigned int> inDim{ 1, 3, 3, 2 };
	vector<unsigned int> outDim{ 1, 2, 2, 2 };
	TensorXF in(inDim);
	TensorXF out(outDim);
	
	std::default_random_engine e;
	std::normal_distribution<float> n(0., 1.);
	for (int i = 0; i < in.size(); ++i)
	{
		in[i] = n(e);
		cout << in[i] << "  ";
	}
	cout <<endl<< "input prepared" << endl;
	out.fillData(0.5f);
	ConvolutionLayer C(2,3);
	ReluLayer R;
	MaxPoolingLayer M;
	M.forward(in, out);
	for (int i = 0; i < out.size(); ++i)
	{
		cout << out[i] << " ";
	}
	system("pause");
	return 0;
}