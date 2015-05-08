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
	void InterlockBackPropInput(const vector<LayerMemoryDescription>& inputDescriptions);
	void InterlockBackPropOutput(
			const vector<LayerMemoryDescription>& outputDescriptions);

	virtual bool Interlocked() const
	{
		return inputInterlocked && outputInterlocked && forwardInputInterlocked;
	}
	;

	vector<LayerMemoryDescription> InForwardPropMemoryDescription() const;
	vector<LayerMemoryDescription> InBackPropMemoryDescription() const;
	vector<LayerMemoryDescription> OutBackPropMemoryDescription() const;

	vector<LayerMemoryDescription> InForwardPropMemoryProposal() const
	{
		return inForwardPropMemoryProposals;
	}
	;

	vector<LayerDataDescription> InForwardPropDataDescription() const
	{
		return inForwardPropDataDescriptions;
	}
	;

	vector<LayerDataDescription> InBackPropDataDescription() const
	{
		return inBackPropDataDescriptions;
	}
	;

	vector<LayerDataDescription> OutBackPropDataDescription() const
	{
		return inForwardPropDataDescriptions; //Must be equal by definition
	}
	;

	vector<LayerMemoryDescription> InBackPropMemoryProposal() const
	{
		return inBackPropMemoryProposals;
	}
	;

	vector<LayerMemoryDescription> OutBackPropMemoryProposal() const
	{
		return outBackPropMemoryProposals;
	}
	;
};
} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_BACKPROPLAYER_H_ */
