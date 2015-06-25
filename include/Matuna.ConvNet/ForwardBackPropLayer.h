/*
 * ForthBackPropLayer.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_FORTHBACKPROPLAYER_H_
#define MATUNA_MATUNA_CONVNET_FORTHBACKPROPLAYER_H_

#include "BackPropLayer.h"
#include "ForwardBackPropLayerConfig.h"

namespace Matuna
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
			MatunaActivationFunction backPropActivation,
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
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_FORTHBACKPROPLAYER_H_ */
