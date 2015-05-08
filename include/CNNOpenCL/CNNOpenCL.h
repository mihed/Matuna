/*
 * CNNOpenCL.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_CNNOPENCL_H_
#define ATML_CNNOPENCL_CNNOPENCL_H_

#include "CNN/CNNConfig.h"
#include "CNN/CNN.h"
#include "OpenCLHelper/OpenCLContext.h"
#include "OpenCLForwardBackPropLayer.h"
#include "StandardOutputLayer.h"

#include <memory>

using namespace std;
using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

class CNNOpenCL: public CNN
{
private:
	shared_ptr<OpenCLContext> context;
	vector<unique_ptr<OpenCLForwardBackPropLayer>> layers;
	unique_ptr<StandardOutputLayer> outputLayer;
public:
	//TODO: Add a vector of contexts
	CNNOpenCL(unique_ptr<OpenCLContext> context, CNNConfig config);
	~CNNOpenCL();

	template<class T>
	void FeedForward(const T* input, int formatIndex, T* output);

	template<class T>
	T CalculateError(const T* propagatedValue, int formatIndex,
			const T* target);

	template<class T>
	void CalculateGradient(const T* input, int formatIndex, T* output);

	template<class T>
	void GetParameters(T* parameters);

	size_t GetParameterCount();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_CNNOPENCL_H_ */
