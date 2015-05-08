/*
 * CNN.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CNN_H_
#define ATML_CNN_CNN_H_

#include "CNNConfig.h"
#include "LayerDescriptions.h"

#include <vector>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

class CNN
{
protected:
	vector<LayerDataDescription> inputDataDescriptions;
	vector<LayerMemoryDescription> inputMemoryDescriptions;

public:
	CNN(const CNNConfig& config);
	virtual ~CNN();

	vector<LayerDataDescription> InputDataDescriptions() const;
	vector<LayerMemoryDescription> InputMemoryDescriptions() const;

	//TODO: make this call safer by using templates
	//virtual void FeedForward(const void* input, int formatIndex, void* output) = 0;
	//virtual double CalculateError(const void* propagatedValue, int formatIndex,
	//		const void* target) = 0;
	//virtual void CalculateGradient(const void* input, int formatIndex,
	//		void* output) = 0;
	//virtual void GetParameters(void* parameters) = 0;
	//virtual size_t GetParameterCount() = 0;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNN_H_ */
