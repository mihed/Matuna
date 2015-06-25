/*
 * ConvNet.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_CONVNET_H_
#define MATUNA_MATUNA_CONVNET_CONVNET_H_

#include "ConvNetConfig.h"
#include "LayerDescriptions.h"

#include <vector>

using namespace std;

namespace Matuna
{
namespace MachineLearning
{

class ConvNet
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
	ConvNet(const ConvNetConfig& config);
	virtual ~ConvNet();

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

#endif /* MATUNA_MATUNA_CONVNET_CONVNET_H_ */
