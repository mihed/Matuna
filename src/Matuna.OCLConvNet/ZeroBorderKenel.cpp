/*
 * ZeroBorderKenel.cpp
 *
 *  Created on: May 23, 2015
 *      Author: Mikael
 */

#include "ZeroBorderKenel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
ZeroBorderKenel<T>::ZeroBorderKenel(int dataWidth, int dataHeight,
		int dataUnits, int borderStartLeft, int borderStartRight,
		int borderStartUp, int borderStartDown, int borderHorizontalSize, int borderVerticalSize, int inputStride,
		int inputMemoryHeight, int inputUnitOffset) :
		dataWidth(dataWidth), dataHeight(dataHeight), borderStartLeft(
				borderStartLeft), borderStartRight(borderStartRight), borderStartUp(
				borderStartUp), borderStartDown(borderStartDown), borderHorizontalSize(borderHorizontalSize),
				borderVerticalSize(borderVerticalSize), inputStride(inputStride), inputMemoryHeight(
				inputMemoryHeight), inputUnitOffset(inputUnitOffset)
{
	stringstream stringStream;

	stringStream << "ZeroBorderKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	useRelaxedMath = false;

	globalWorkSize.push_back(dataUnits);

	kernelName = "ZeroBorderKernel";
}

template<class T>
ZeroBorderKenel<T>::~ZeroBorderKenel()
{

}

template<class T>
void ZeroBorderKenel<T>::InitializeCompilerOptions()
{
	stringstream stringStream;

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING=" << (inputStride * inputMemoryHeight) << " ";
	stringStream << "-D" << "BORDER_START_LEFT=" << borderStartLeft << " ";
	stringStream << "-D" << "BORDER_START_RIGHT=" << borderStartRight << " ";
	stringStream << "-D" << "BORDER_START_UP=" << borderStartUp << " ";
	stringStream << "-D" << "BORDER_START_DOWN=" << borderStartDown << " ";
	stringStream << "-D" << "BORDER_LIMIT_LEFT=" << (borderStartLeft + borderHorizontalSize - 1) << " ";
	stringStream << "-D" << "BORDER_LIMIT_RIGHT=" << (borderStartRight + borderHorizontalSize - 1) << " ";
	stringStream << "-D" << "BORDER_LIMIT_UP=" << (borderStartUp + borderVerticalSize - 1) << " ";
	stringStream << "-D" << "BORDER_LIMIT_DOWN=" << (borderStartDown + borderVerticalSize - 1) << " ";
	stringStream << "-D" << "BORDER_SIZE_HORIZONTAL=" << borderHorizontalSize << " ";
	stringStream << "-D" << "BORDER_SIZE_VERTICAL=" << borderVerticalSize << " ";
	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputUnitOffset << " ";
	stringStream << "-D" << "INPUT_DATA_WIDTH=" << dataWidth << " ";
	stringStream << "-D" << "INPUT_DATA_HEIGHT=" << dataHeight << " ";
	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();
}

template<class T>
void ZeroBorderKenel<T>::SetInputOutput(OCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ZeroBorderKenel<T>::SetUseRelaxedMath(bool value)
{
	this->useRelaxedMath = value;
}

template<class T>
string ZeroBorderKenel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string ZeroBorderKenel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> ZeroBorderKenel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"ZeroBorderKernel.cl")));
	return result;
}

template<class T>
string ZeroBorderKenel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& ZeroBorderKenel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ZeroBorderKenel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class ZeroBorderKenel<cl_float> ;
template class ZeroBorderKenel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
