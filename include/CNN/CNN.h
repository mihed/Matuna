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
private:
	vector<LayerDataDescription> inputForwardDataDescriptions;
	vector<LayerMemoryDescription> inputForwardMemoryDescriptions;
	vector<LayerMemoryDescription> outputBackMemoryDescriptions;

	vector<LayerMemoryDescription> outputForwardMemoryDescriptions;
	vector<LayerDataDescription> outputForwardDataDescriptions;


	bool inputInterlocked;
	bool outputInterlocked;
	bool outputDataInterlocked;
	bool outputBackInterlocked;

public:
	CNN(const CNNConfig& config);
	virtual ~CNN();

	void InterlockForwardPropInput(
			const vector<LayerMemoryDescription>& inputDescriptions);

	void InterlockForwardPropOutput(
			const vector<LayerMemoryDescription>& outputDescriptions);

	void InterlockBackPropOutput(
		const vector<LayerMemoryDescription>& outputDescriptions);

	void InterlockForwardPropDataOutput(
			const vector<LayerDataDescription>& outputDescriptions);

	bool Interlocked() const;

	vector<LayerDataDescription> InputForwardDataDescriptions() const;
	vector<LayerMemoryDescription> InputForwardMemoryDescriptions() const;
	vector<LayerMemoryDescription> OutputBackMemoryDescriptions() const;

	vector<LayerDataDescription> OutputForwardDataDescriptions() const;
	vector<LayerMemoryDescription> OutputForwardMemoryDescriptions() const;

};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNN_H_ */
