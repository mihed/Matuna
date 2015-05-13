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

	vector<LayerMemoryDescription> outForwardPropMemoryDescriptions;

protected:
	vector<LayerDataDescription> outForwardPropDataDescriptions; //Must be set equal to inBackPropDataDescription
	vector<LayerMemoryDescription> outForwardPropMemoryProposals; //Must be set inside constructor for derived classes

public:
	ForwardBackPropLayer(
			const vector<LayerDataDescription>& inputLayerDescriptions,
			const ForwardBackPropLayerConfig* config);
	virtual ~ForwardBackPropLayer();

	void InterlockForwardPropOutput(
			const vector<LayerMemoryDescription>& outputDescriptions);

	virtual bool Interlocked() const override;

	vector<LayerMemoryDescription> OutForwardPropMemoryDescriptions() const;
	vector<LayerDataDescription> OutForwardPropDataDescriptions() const;
	vector<LayerMemoryDescription> OutForwardPropMemoryProposals() const;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_FORTHBACKPROPLAYER_H_ */
