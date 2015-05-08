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
#include "OpenCLHelper/OpenCLContext.h"

using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

class PerceptronLayer: public OpenCLForwardBackPropLayer
{
public:
	PerceptronLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
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
