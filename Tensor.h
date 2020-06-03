#pragma once

#include<vector>
#include <algorithm>
#include <numeric>

template<typename T=float>
class Tensor
{
private:
	std::vector<unsigned int> dimensions;
	T* dataOffset;

	inline const unsigned int computeIndex(const std::vector<unsigned int>& indices) const
	{
		unsigned int index = indices[0];

		for (unsigned int i = 1; i < indices.size(); i++)
			index = index * dimensions[i] + indices[i];
		return index;
	}

public:
	Tensor(){ dimensions = std::vector<int>{}; dataOffset = std::nullptr_t; }
	Tensor(const std::vector<unsigned int> _dimensions) :dimensions(_dimensions)
		, dataOffset(new T[this->size()])  {}
	~Tensor()  { delete dataOffset; }

//	inline const T* data() const{ return dataOffset; }
	inline  T* data() const{ return dataOffset; }
	bool reshape(std::vector<unsigned int> _dimensions) {
		if (std::accumulate(_dimensions.begin(), _dimensions.end(), 1, multiplies<int>())
			== std::accumulate(dimensions.begin(), dimensions.end(), 1, multiplies<int>()
			))
		{
			dimensions = _dimensions;
			return true;
			}
		return false;
	}
	inline const std::size_t  size() const
	{
		std::size_t size= 1;
		for (auto i : dimensions)
			size *= i;
		return size;
	}

    std::vector<unsigned int> dim() const { return this->dimensions; }
	void fillData(const T value)
	{
		std::size_t size = this->size();
		T* tmp = this->data();
		for (std::size_t i=0; i < size; ++i)
		{
			*tmp++ = value;
		}
	}
	void cloneTo(const Tensor<T> target)
	{
		target.dimensions = this->dimensions;
		const std::size_t len = sizeof(T)*this->size();
		memcpy(target.data(), this->data(), len);
	}
	inline  T& operator[]( std::size_t index) const { return *(dataOffset + index); }
//	inline const T& operator()(const std::vector<unsigned int>& indices) const { return (*this)[computeIndex(indices)]; }
	inline T& operator()( std::vector<unsigned int>& indices) { return (*this)[computeIndex(indices)]; }
};
using TensorXF = Tensor<float>;