/*
 * StandardOutputLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef CNNOPENCL_STANDARDOUTPUTLAYER_H_
#define CNNOPENCL_STANDARDOUTPUTLAYER_H_

#include "CNN/OutputLayer.h"
#include "OpenCLHelper/OpenCLDevice.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include <memory>

using namespace std;
using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class StandardOutputLayer final: public OutputLayer
{
private:
	shared_ptr<OpenCLContext> context;

public:
	StandardOutputLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			ATMLActivationFunction backPropActivation,
			const OutputLayerConfig* outputLayerConfig);
	~StandardOutputLayer();

	virtual void InterlockFinalized() override;

	void EnqueueBackPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* deltaOutput, bool blocking = true);
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_STANDARDOUTPUTLAYER_H_ */
