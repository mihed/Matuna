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

	virtual void EnqueueForwardPropagation(
			shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> output) = 0;

	virtual void EnqueueBackPropagation(shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> delta,
			shared_ptr<OpenCLMemory> deltaOutput) = 0;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_OPENCLFORWARDBACKPROPLAYER_H_ */
