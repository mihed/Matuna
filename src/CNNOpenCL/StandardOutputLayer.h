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
			const OutputLayerConfig* outputLayerConfig);
	~StandardOutputLayer();

	virtual void InterlockFinalized() override;

	void EnqueueBackPropagation(shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> target,
			shared_ptr<OpenCLMemory> deltaOutput);
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_STANDARDOUTPUTLAYER_H_ */
