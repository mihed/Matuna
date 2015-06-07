/*
 * LayerDescriptions.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CONVNET_LAYERDESCRIPTIONS_H_
#define MATUNA_CONVNET_LAYERDESCRIPTIONS_H_

namespace Matuna
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

	int TotalUnits() const
	{
		return Width * Height * Units;
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

	int TotalMemory() const
	{
		return Width * Height * Units;
	}
	;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CONVNET_LAYERDESCRIPTIONS_H_ */
