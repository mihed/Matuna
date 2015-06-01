/*
 * StandardOutputLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef CNNOPENCL_STANDARDOUTPUTLAYER_H_
#define CNNOPENCL_STANDARDOUTPUTLAYER_H_

#include "CNN/OutputLayer.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "OpenCLHelper/OpenCLContext.h"
#include "OpenCLHelper/OpenCLDevice.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include "OutputKernel.h"
#include "ErrorKernel.h"
#include "ImageErrorKernel.h"
#include "ImageOutputKernel.h"
#include <memory>
#include <unordered_map>

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
	unordered_map<OpenCLDevice*, unique_ptr<OutputKernel<T>>> deviceAndOutputKernels;
	unordered_map<OpenCLDevice*, unique_ptr<ErrorKernel<T>>> deviceAndErrorKernels;
	unordered_map<OpenCLDevice*, unique_ptr<ImageErrorKernel<T>>> deviceAndImageErrorKernels;
	unordered_map<OpenCLDevice*, unique_ptr<ImageOutputKernel<T>>> deviceAndImageOutputKernels;

	bool useImage;

	StandardOutputLayerConfig config;
	LayerDataDescription inputDescription;

public:
	StandardOutputLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			ATMLActivationFunction backPropActivation,
			const StandardOutputLayerConfig* outputLayerConfig);
	~StandardOutputLayer();

	virtual void InterlockFinalized() override;

	void EnqueueBackPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* target,
			OpenCLMemory* deltaOutput, bool blocking = true);

	T CalculateError(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* target);

private:
	void InitializeErrorKernel();
	void InitializeImageErrorKernel();
	void InitializeOutputKernel();
	void InitializeImageOutputKernel();
};

}
/* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_STANDARDOUTPUTLAYER_H_ */
