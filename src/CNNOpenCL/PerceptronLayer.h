/*
 * PerceptronLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef CNNOPENCL_PERCEPTRONLAYER_H_
#define CNNOPENCL_PERCEPTRONLAYER_H_

#include "OpenCLForwardBackPropLayer.h"
#include "ForwardPerceptronKernel.h"
#include "CNN/PerceptronLayerConfig.h"
#include "OpenCLHelper/OpenCLContext.h"

#include <memory>

using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class PerceptronLayer: public OpenCLForwardBackPropLayer<T>
{

private:
	unique_ptr<ForwardPerceptronKernel> forwardPerceptronKernel;
public:
	PerceptronLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			const PerceptronLayerConfig* config);
	virtual ~PerceptronLayer();

	virtual void InterlockFinalized() override;

	virtual void EnqueueForwardPropagation(
			shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> output) override;

	virtual void EnqueueBackPropagation(shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> delta,
			shared_ptr<OpenCLMemory> deltaOutput) override;

private:
	void InitializeNormalPerceptron();
	void InitializeImagePerceptron();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_PERCEPTRONLAYER_H_ */
