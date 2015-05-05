/*
 * OpenCLForwardBackPropLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_OPENCLFORWARDBACKPROPLAYER_H_
#define ATML_CNNOPENCL_OPENCLFORWARDBACKPROPLAYER_H_

#include "CNN/ForwardBackPropLayer.h"
#include "OpenCLHelper/OpenCLDevice.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include <memory>

using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

class OpenCLForwardBackPropLayer: public ForwardBackPropLayer
{
public:
	OpenCLForwardBackPropLayer(
			const LayerDataDescription& inputLayerDescription,
			const ForwardBackPropLayerConfig* config);
	~OpenCLForwardBackPropLayer();

	virtual void EnqueueForwardPropagation(shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> output) = 0;

	virtual void EnqueueBackPropagation(shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> delta,
			shared_ptr<OpenCLMemory> deltaOutput) = 0;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_OPENCLFORWARDBACKPROPLAYER_H_ */
