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
	vector<LayerDataDescription> inputDataDescriptions;
	vector<LayerMemoryDescription> inputMemoryDescriptions;
	vector<LayerMemoryDescription> inputMemoryProposals;

	vector<LayerMemoryDescription> outputMemoryDescriptions;
	vector<LayerDataDescription> outputDataDescriptions;

	bool inputInterlocked;
	bool outputInterlocked;
	bool outputDataInterlocked;

public:
	CNN(const CNNConfig& config);
	virtual ~CNN();

	void InterlockForwardPropInput(
			const vector<LayerMemoryDescription>& inputDescriptions);

	void InterlockForwardPropOutput(
			const vector<LayerMemoryDescription>& outputDescriptions);

	void InterlockForwardPropDataOutput(
			const vector<LayerDataDescription>& outputDescriptions);

	bool Interlocked() const;

	vector<LayerDataDescription> InputDataDescriptions() const;
	vector<LayerMemoryDescription> InputMemoryDescriptions() const;
	vector<LayerMemoryDescription> InputMemoryProposals() const;

	vector<LayerDataDescription> OutputDataDescriptions() const;
	vector<LayerMemoryDescription> OutputMemoryDescriptions() const;

};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNN_H_ */
