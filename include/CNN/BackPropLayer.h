/*
 * BackPropLayer.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_BACKPROPLAYER_H_
#define ATML_CNN_BACKPROPLAYER_H_

#include "Layer.h"
#include "LayerDescriptions.h"
#include <vector>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

class BackPropLayer: public Layer
{
private:
	bool forwardInputInterlocked;
	bool inputInterlocked;
	bool outputInterlocked;

	vector<LayerDataDescription> inForwardPropDataDescriptions;

	vector<LayerMemoryDescription> inBackPropMemoryDescriptions;
	vector<LayerMemoryDescription> outBackPropMemoryDescriptions;

	//Since a back prop layer requires input of a previous forward layer.
	vector<LayerMemoryDescription> inForwardPropMemoryDescriptions;

protected:
	vector<LayerDataDescription> inBackPropDataDescriptions; //Must be set inside constructor for derived classes

	vector<LayerMemoryDescription> inBackPropMemoryProposals; //Must be set inside constructor for derived classes
	vector<LayerMemoryDescription> outBackPropMemoryProposals; //Must be set inside constructor for derived classes

	//Since a back prop layer requires input of a previous forward layer.
	vector<LayerMemoryDescription> inForwardPropMemoryProposals; //Must be set inside constructor for derived classes

public:
	BackPropLayer(const vector<LayerDataDescription>& inputLayerDescriptions);
	virtual ~BackPropLayer();

	void InterlockForwardPropInput(
			const vector<LayerMemoryDescription>& inputDescriptions);
	void InterlockBackPropInput(
			const vector<LayerMemoryDescription>& inputDescriptions);
	void InterlockBackPropOutput(
			const vector<LayerMemoryDescription>& outputDescriptions);

	virtual bool Interlocked() const;

	virtual void InterlockFinalized() = 0;

	vector<LayerMemoryDescription> InForwardPropMemoryDescriptions() const;
	vector<LayerMemoryDescription> InBackPropMemoryDescriptions() const;
	vector<LayerMemoryDescription> OutBackPropMemoryDescriptions() const;
	vector<LayerMemoryDescription> InForwardPropMemoryProposals() const;
	vector<LayerDataDescription> InForwardPropDataDescriptions() const;
	vector<LayerDataDescription> InBackPropDataDescriptions() const;
	vector<LayerDataDescription> OutBackPropDataDescriptions() const;
	vector<LayerMemoryDescription> InBackPropMemoryProposals() const;
	vector<LayerMemoryDescription> OutBackPropMemoryProposals() const;
};
} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_BACKPROPLAYER_H_ */
