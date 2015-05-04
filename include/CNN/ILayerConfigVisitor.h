/*
 * ILayerConfigVisitor.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef CNN_ILAYERCONFIGVISITOR_H_
#define CNN_ILAYERCONFIGVISITOR_H_

namespace ATML
{
namespace MachineLearning
{

class CNNConfig;

class ILayerConfigVisitor
{

public:

	virtual ~ILayerConfigVisitor()
	{
	}
	;

	virtual void Visit(const CNNConfig* const cnnConfig) = 0;

};

} /* ATML */
} /* MachineLearning */

#endif /* CNN_ILAYERCONFIGVISITOR_H_ */
