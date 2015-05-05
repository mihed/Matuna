/*
 * PerceptronLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef CNNOPENCL_PERCEPTRONLAYER_H_
#define CNNOPENCL_PERCEPTRONLAYER_H_

#include "OpenCLForwardBackPropLayer.h"
#include "CNN/PerceptronLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

class PerceptronLayer: public OpenCLForwardBackPropLayer
{
public:
	PerceptronLayer(const LayerDataDescription& inputLayerDescription,
			const PerceptronLayerConfig* config);
	~PerceptronLayer();

	virtual void EnqueueForwardPropagation(
			shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> output) override;

	virtual void EnqueueBackPropagation(shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> delta,
			shared_ptr<OpenCLMemory> deltaOutput) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_PERCEPTRONLAYER_H_ */
