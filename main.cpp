#include<iostream>
#include<vector>
#include <random>
#include"Tensor.h"
#include"ActivationLayer.h"
#include"ConvolutionLayer.h"
#include"PoolingLayer.h"
#include"SoftmaxLayer.h"
#include"LossLayer.h"

using namespace std;
int main()
{
	vector<unsigned int> inDim{ 1, 10};
	vector<unsigned int> preDim{ 1 };
	TensorXF in(inDim);
	Tensor<unsigned> pre(preDim);
	std::default_random_engine e;
	std::normal_distribution<float> n(0., 1.);
	for (int i = 0; i < in.size(); ++i)
	{
		in[i] = n(e);
		cout << in[i] << "  ";
	}
	cout <<endl<< "input prepared" << endl;
	pre.fillData(3);
	SoftmaxLayer S(1, 10);
	CrossEntropyLossLayer L(1, 10);
	auto out = S.forward(in);
	L.forward(out, pre);
	auto out2 = L.backward(out, pre);
	S.backward(out2);
	system("pause");
	return 0;
}

