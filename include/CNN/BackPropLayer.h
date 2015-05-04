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

	LayerDataDescription inForwardPropDataDescription;

	LayerMemoryDescription inBackPropMemoryDescription;
	LayerMemoryDescription outBackPropMemoryDescription;

	//Since a back prop layer requires input of a previous forward layer.
	LayerMemoryDescription inForwardPropMemoryDescription;

protected:
	LayerDataDescription inBackPropDataDescription; //Must be set inside constructor for derived classes

	LayerMemoryDescription inBackPropMemoryProposal; //Must be set inside constructor for derived classes
	LayerMemoryDescription outBackPropMemoryProposal; //Must be set inside constructor for derived classes

	//Since a back prop layer requires input of a previous forward layer.
	LayerMemoryDescription inForwardPropMemoryProposal; //Must be set inside constructor for derived classes

public:
	BackPropLayer(const LayerDataDescription& inputLayerDescription);
	virtual ~BackPropLayer();

	void InterlockForwardPropInput(
			const LayerMemoryDescription& inputDescription);
	void InterlockBackPropInput(const LayerMemoryDescription& inputDescription);
	void InterlockBackPropOutput(
			const LayerMemoryDescription& outputDescription);

	virtual bool Interlocked() const
	{
		return inputInterlocked && outputInterlocked && forwardInputInterlocked;
	}
	;

	LayerMemoryDescription InForwardPropMemoryDescription() const;
	LayerMemoryDescription InBackPropMemoryDescription() const;
	LayerMemoryDescription OutBackPropMemoryDescription() const;

	LayerMemoryDescription InForwardPropMemoryProposal() const
	{
		return inForwardPropMemoryProposal;
	}
	;

	LayerDataDescription InForwardPropDataDescription() const
	{
		return inForwardPropDataDescription;
	}
	;

	LayerDataDescription InBackPropDataDescription() const
	{
		return inBackPropDataDescription;
	}
	;
	LayerDataDescription OutBackPropDataDescription() const
	{
		return inForwardPropDataDescription; //Must be equal by definition
	}
	;
	LayerMemoryDescription InBackPropMemoryProposal() const
	{
		return inBackPropMemoryProposal;
	}
	;
	LayerMemoryDescription OutBackPropMemoryProposal() const
	{
		return outBackPropMemoryProposal;
	}
	;
};
} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_BACKPROPLAYER_H_ */
