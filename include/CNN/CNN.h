/*
 * CNN.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNN_CNN_H_
#define MATUNA_CNN_CNN_H_

#include "CNNConfig.h"
#include "LayerDescriptions.h"

#include <vector>

using namespace std;

namespace Matuna
{
namespace MachineLearning
{

class CNN
{
private:
	vector<LayerDataDescription> inputForwardDataDescriptions;
	vector<LayerMemoryDescription> inputForwardMemoryDescriptions;

	vector<LayerMemoryDescription> outputForwardMemoryDescriptions;
	vector<LayerDataDescription> outputForwardDataDescriptions;

	vector<LayerMemoryDescription> outputBackMemoryDescriptions;
	vector<LayerDataDescription> outputBackDataDescriptions;

	bool inputInterlocked;
	bool outputInterlocked;
	bool outputDataInterlocked;
	bool outputBackInterlocked;
	bool outputBackDataInterlocked;

public:
	CNN(const CNNConfig& config);
	virtual ~CNN();

	void InterlockForwardPropInput(
			const vector<LayerMemoryDescription>& inputDescriptions);

	void InterlockForwardPropOutput(
			const vector<LayerMemoryDescription>& outputDescriptions);

	void InterlockBackPropOutput(
		const vector<LayerMemoryDescription>& outputDescriptions);

	void InterlockBackPropDataOutput(
		const vector<LayerDataDescription>& outputDescriptions);

	void InterlockForwardPropDataOutput(
			const vector<LayerDataDescription>& outputDescriptions);

	bool Interlocked() const;

	vector<LayerDataDescription> InputForwardDataDescriptions() const;
	vector<LayerMemoryDescription> InputForwardMemoryDescriptions() const;

	vector<LayerMemoryDescription> OutputBackMemoryDescriptions() const;
	vector<LayerDataDescription> OutputBackDataDescriptions() const;

	vector<LayerDataDescription> OutputForwardDataDescriptions() const;
	vector<LayerMemoryDescription> OutputForwardMemoryDescriptions() const;

};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNN_CNN_H_ */
