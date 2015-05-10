/*
 * OpenCLForwardBackPropLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_OPENCLFORWARDBACKPROPLAYER_H_
#define ATML_CNNOPENCL_OPENCLFORWARDBACKPROPLAYER_H_

#include "CNN/ForwardBackPropLayer.h"
#include "OpenCLHelper/OpenCLContext.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include <memory>

using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class OpenCLForwardBackPropLayer: public ForwardBackPropLayer
{
protected:
	shared_ptr<OpenCLContext> context;

public:
	OpenCLForwardBackPropLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			const ForwardBackPropLayerConfig* config);
	virtual ~OpenCLForwardBackPropLayer();

	virtual void EnqueueForwardPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* output, bool blocking =
					true) = 0;

	virtual void EnqueueBackPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* deltaOutput, bool blocking = true) = 0;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_OPENCLFORWARDBACKPROPLAYER_H_ */
