/*
 * LayerDescriptions.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_LAYERDESCRIPTIONS_H_
#define ATML_CNN_LAYERDESCRIPTIONS_H_

namespace ATML
{
namespace MachineLearning
{

class LayerDataDescription
{
public:
	int Width;
	int Height;
	int Units;

public:
	LayerDataDescription()
	{
		Width = -1;
		Height = -1;
		Units = -1;
	}
	;

	~LayerDataDescription()
	{

	}
	;
};

class LayerMemoryDescription
{
public:
	int Width;
	int Height;
	int Units;
	int WidthOffset;
	int HeightOffset;
	int UnitOffset;
public:
	LayerMemoryDescription()
	{
		Width = -1;
		Height = -1;
		Units = -1;
		WidthOffset = -1;
		HeightOffset = -1;
		UnitOffset = -1;
	}
	;

	~LayerMemoryDescription()
	{

	}
	;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_LAYERDESCRIPTIONS_H_ */
