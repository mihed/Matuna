/*
 * ConvolutionLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef CNNOPENCL_CONVOLUTIONLAYER_H_
#define CNNOPENCL_CONVOLUTIONLAYER_H_

#include "OpenCLForwardBackPropLayer.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "OpenCLHelper/OpenCLContext.h"

using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class ConvolutionLayer: public OpenCLForwardBackPropLayer<T>
{
public:
	ConvolutionLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			const ConvolutionLayerConfig* config);
	virtual ~ConvolutionLayer();

	virtual void InterlockFinalized() override;

	virtual void EnqueueForwardPropagation(
			shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> output) override;

	virtual void EnqueueBackPropagation(shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> delta,
			shared_ptr<OpenCLMemory> deltaOutput) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_CONVOLUTIONLAYER_H_ */
