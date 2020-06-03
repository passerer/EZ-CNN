#include<iostream>
#include<vector>
#include <random>
#include"Tensor.h"
#include"ActivationLayer.h"

using namespace std;
int main()
{
	vector<unsigned int> dim{ 2, 1, 2, 2 };
	TensorXF in(dim);
	TensorXF out(dim);
	default_random_engine e;
	uniform_real_distribution<float> u(-0.1, 1.0);
	for (int i = 0; i < in.size(); ++i)
	{
		in[i] = u(e);
	}
	out.fillData(0.f);
	SigmodLayer si;
	si.forward(in, out);
	for (int i = 0; i < out.size(); ++i)
	{
		cout << out[i] << "  ";
	}
	system("pause");
	return 0;
}