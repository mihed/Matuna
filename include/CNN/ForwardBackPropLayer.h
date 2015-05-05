/*
 * ForthBackPropLayer.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_FORTHBACKPROPLAYER_H_
#define ATML_CNN_FORTHBACKPROPLAYER_H_

#include "BackPropLayer.h"
#include "ForwardBackPropLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

class ForwardBackPropLayer: public BackPropLayer
{
private:
	bool outputInterlocked;

	LayerMemoryDescription outForwardPropMemoryDescription;

protected:
	LayerDataDescription outForwardPropDataDescription; //Must be set equal to inBackPropDataDescription
	LayerMemoryDescription outForwardPropMemoryProposal; //Must be set inside constructor for derived classes

public:
	ForwardBackPropLayer(const LayerDataDescription& inputLayerDescription,
			const ForwardBackPropLayerConfig* config);
	virtual ~ForwardBackPropLayer();

	void InterlockForwardPropOutput(
			const LayerMemoryDescription& outputDescription);

	virtual bool Interlocked() const override
	{
		return outputInterlocked
				&& BackPropLayer::Interlocked();
	}
	;

	LayerMemoryDescription OutForwardPropMemoryDescription() const;

	LayerDataDescription OutForwardPropDataDescription() const
	{
		return outForwardPropDataDescription;
	}
	;
	LayerMemoryDescription OutForwardPropMemoryProposal() const
	{
		return outForwardPropMemoryProposal;
	}
	;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_FORTHBACKPROPLAYER_H_ */
