/*
 * StandardOutputLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef CNNOCL_STANDARDOUTPUTLAYER_H_
#define CNNOCL_STANDARDOUTPUTLAYER_H_

#include "CNN/OutputLayer.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "OCLHelper/OCLContext.h"
#include "OCLHelper/OCLDevice.h"
#include "OCLHelper/OCLMemory.h"
#include "OutputKernel.h"
#include "ErrorKernel.h"
#include "ImageErrorKernel.h"
#include "ImageOutputKernel.h"
#include <memory>
#include <unordered_map>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class StandardOutputLayer final: public OutputLayer
{
private:
	shared_ptr<OCLContext> context;
	unordered_map<OCLDevice*, unique_ptr<OutputKernel<T>>> deviceAndOutputKernels;
	unordered_map<OCLDevice*, unique_ptr<ErrorKernel<T>>> deviceAndErrorKernels;
	unordered_map<OCLDevice*, unique_ptr<ImageErrorKernel<T>>> deviceAndImageErrorKernels;
	unordered_map<OCLDevice*, unique_ptr<ImageOutputKernel<T>>> deviceAndImageOutputKernels;

	bool useImage;

	StandardOutputLayerConfig config;
	LayerDataDescription inputDescription;

public:
	StandardOutputLayer(shared_ptr<OCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const StandardOutputLayerConfig* outputLayerConfig);
	~StandardOutputLayer();

	virtual void InterlockFinalized() override;

	void EnqueueBackPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* target,
			OCLMemory* deltaOutput, bool blocking = true);

	T CalculateError(OCLDevice* device,
			int queueIndex, OCLMemory* previousInput, OCLMemory* target);

private:
	void InitializeErrorKernel();
	void InitializeImageErrorKernel();
	void InitializeOutputKernel();
	void InitializeImageOutputKernel();
};

}
/* namespace MachineLearning */
} /* namespace Matuna */

#endif /* CNNOCL_STANDARDOUTPUTLAYER_H_ */
