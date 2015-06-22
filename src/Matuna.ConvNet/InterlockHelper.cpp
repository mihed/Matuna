/*
 * InterlockHelper.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#include "InterlockHelper.h"
#include "LayerDescriptions.h"
#include <exception>
#include <stdexcept>

using namespace std;

namespace Matuna
{
namespace MachineLearning
{

LayerMemoryDescription InterlockHelper::CalculateCompatibleMemory(
		const LayerMemoryDescription& left, const LayerMemoryDescription& right)
{
	if (!IsCompatible(right, left))
		throw runtime_error(
				"The memories are incompatible and no comptabile memory can be created.");

	//Don't forget that this function is only intended between two layers,
	//Meaning that the actual data is the same. We just need to increase the size so that it fit both descriptions.

	int widthOffset;
	if (left.WidthOffset < right.WidthOffset)
		widthOffset = right.WidthOffset;
	else
		widthOffset = left.WidthOffset;

	int heightOffset;
	if (left.HeightOffset < right.HeightOffset)
		heightOffset = right.HeightOffset;
	else
		heightOffset = left.HeightOffset;

	int unitOffset;
	if (left.UnitOffset < right.UnitOffset)
		unitOffset = right.UnitOffset;
	else
		unitOffset = left.UnitOffset;

	//Now when we have compatible offsets. We need to make sure that we keep the offset - dimension of the greatest.

	int effectiveWidth;
	if (left.Width - left.WidthOffset < right.Width - right.WidthOffset)
		effectiveWidth = right.Width - right.WidthOffset;
	else
		effectiveWidth = left.Width - left.WidthOffset;

	int effectiveHeight;
	if (left.Height - left.HeightOffset < right.Height - right.HeightOffset)
		effectiveHeight = right.Height - right.HeightOffset;
	else
		effectiveHeight = left.Height - left.HeightOffset;

	int effectiveUnits;
	if (left.Units - left.UnitOffset < right.Units - right.UnitOffset)
		effectiveUnits = right.Units - right.UnitOffset;
	else
		effectiveUnits = left.Units - left.UnitOffset;

	LayerMemoryDescription result;
	result.Height = effectiveHeight + heightOffset;
	result.Width = effectiveWidth + widthOffset;
	result.Units = effectiveUnits + unitOffset;
	result.WidthOffset = widthOffset;
	result.HeightOffset = heightOffset;
	result.UnitOffset = unitOffset;

	return result;
}

vector<LayerMemoryDescription> InterlockHelper::CalculateCompatibleMemory(
		const vector<LayerMemoryDescription>& left,
		const vector<LayerMemoryDescription>& right)
{
	vector<LayerMemoryDescription> result;
	auto count = left.size();

	if (count != right.size())
		throw invalid_argument("The vector size does not match");

	for (size_t i = 0; i < count; i++)
		result.push_back(CalculateCompatibleMemory(left[i], right[i]));

	return result;
}

bool InterlockHelper::IsCompatible(const LayerMemoryDescription& right,
		const LayerMemoryDescription& left)
{
	if (right.Height <= 0)
		throw runtime_error("Invalid memory");
	if (left.Height <= 0)
		throw runtime_error("Invalid memory");
	if (right.Width <= 0)
		throw runtime_error("Invalid memory");
	if (left.Width <= 0)
		throw runtime_error("Invalid memory");
	if (right.Units <= 0)
		throw runtime_error("Invalid memory");
	if (left.Units <= 0)
		throw runtime_error("Invalid memory");
	if (left.WidthOffset < 0)
		throw runtime_error("Invalid memory");
	if (right.WidthOffset < 0)
		throw runtime_error("Invalid memory");
	if (left.HeightOffset < 0)
		throw runtime_error("Invalid memory");
	if (right.HeightOffset < 0)
		throw runtime_error("Invalid memory");
	if (left.UnitOffset < 0)
		throw runtime_error("Invalid memory");
	if (right.UnitOffset < 0)
		throw runtime_error("Invalid memory");

	if (right.UnitOffset > right.Units)
		return false;
	if (right.WidthOffset > right.Width)
		return false;
	if (right.HeightOffset > right.Height)
		return false;

	if (left.UnitOffset > left.Units)
		return false;
	if (left.WidthOffset > left.Width)
		return false;
	if (left.HeightOffset > left.Height)
		return false;

	//Since we don't allow step-size in the memory atm. All other memory is compatible

	return true;
}

bool InterlockHelper::IsCompatible(const vector<LayerMemoryDescription>& right,
		const vector<LayerMemoryDescription>& left)
{
	auto count = left.size();
	if (count != right.size())
		throw invalid_argument("The vector size does not match");

	for (size_t i = 0; i < count; i++)
		if (!IsCompatible(left[i], right[i]))
			return false;

	return true;
}

bool InterlockHelper::IsCompatible(const LayerDataDescription& data,
		const LayerMemoryDescription& memory)
{
	if (data.Width <= 0)
		throw runtime_error("Invalid data");
	if (data.Height <= 0)
		throw runtime_error("Invalid data");
	if (data.Units <= 0)
		throw runtime_error("Invalid data");

	if (memory.Width <= 0)
		throw runtime_error("Invalid memory");
	if (memory.Height <= 0)
		throw runtime_error("Invalid memory");
	if (memory.Units <= 0)
		throw runtime_error("Invalid memory");
	if (memory.UnitOffset < 0)
		throw runtime_error("Invalid memory");
	if (memory.WidthOffset < 0)
		throw runtime_error("Invalid memory");
	if (memory.HeightOffset < 0)
		throw runtime_error("Invalid memory");

	if (data.Width > memory.Width - memory.WidthOffset)
		return false;
	if (data.Height > memory.Height - memory.HeightOffset)
		return false;
	if (data.Units > memory.Units - memory.UnitOffset)
		return false;

	//Since we don't allow step-size in the memory atm. All other memory is compatible

	return true;
}

bool InterlockHelper::IsCompatible(const vector<LayerDataDescription>& data,
		const vector<LayerMemoryDescription>& memory)
{
	auto count = memory.size();
	if (count != data.size())
		throw invalid_argument("The vector size does not match");

	for (size_t i = 0; i < count; i++)
		if (!IsCompatible(data[i], memory[i]))
			return false;

	return true;
}

bool InterlockHelper::MemoryEquals(const LayerMemoryDescription& right,
		const LayerMemoryDescription& left)
{
	if (right.Width != left.Width)
		return false;
	if (right.Height != left.Height)
		return false;
	if (right.Units != left.Units)
		return false;
	if (right.WidthOffset != left.WidthOffset)
		return false;
	if (right.HeightOffset != left.HeightOffset)
		return false;
	if (right.UnitOffset != left.UnitOffset)
		return false;

	return true;
}

bool InterlockHelper::MemoryEquals(const vector<LayerMemoryDescription>& right,
		const vector<LayerMemoryDescription>& left)
{
	auto count = left.size();
	if (count != right.size())
		throw invalid_argument("The vector size does not match");

	for (size_t i = 0; i < count; i++)
		if (!MemoryEquals(left[i], right[i]))
			return false;

	return true;
}

bool InterlockHelper::DataEquals(const LayerDataDescription& right,
		const LayerDataDescription& left)
{
	if (right.Width != left.Width)
		return false;
	if (right.Height != left.Height)
		return false;
	if (right.Units != left.Units)
		return false;

	return true;
}

bool InterlockHelper::DataEquals(const vector<LayerDataDescription>& right,
		const vector<LayerDataDescription>& left)
{
	auto count = left.size();
	if (count != right.size())
		throw invalid_argument("The vector size does not match");

	for (size_t i = 0; i < count; i++)
		if (!DataEquals(left[i], right[i]))
			return false;

	return true;
}

} /* namespace MachineLearning */
} /* namespace Matuna */
