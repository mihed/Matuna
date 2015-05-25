/*
 * MultiplyAllUnitsKernel.h
 *
 *  Created on: May 24, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_MULTIPLYALLUNITSKERNEL_H_
#define ATML_CNNOPENCL_MULTIPLYALLUNITSKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "CNN/ATMLActivationFunctionEnum.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace ATML::Helper;
using namespace std;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class MultiplyAllUnitsKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	bool useRelaxedMath;
	bool useConstantInput;
	bool useConstantInputDelta;
	ATMLActivationFunction activationFunction;

	int globalWidth;
	int globalHeight;
	int globalUnits;
	int inputDeltaStride;
	int outputStride;
	int inputStride;
	int inputDeltaWidthOffset;
	int inputDeltaHeightOffset;
	int outputWidthOffset;
	int outputHeightOffset;
	int outputUnitOffset;
	int inputWidthOffset;
	int inputHeightOffset;
	int inputUnitOffset;
	int outputMemoryHeight;
	int inputMemoryHeight;

public:
	MultiplyAllUnitsKernel(int globalWidth, int globalHeight, int globalUnits,
			int inputDeltaStride, int outputStride, int inputStride,
			int inputDeltaWidthOffset, int inputDeltaHeightOffset,
			int outputWidthOffset, int outputHeightOffset, int outputUnitOffset,
			int inputWidthOffset, int inputHeightOffset, int inputUnitOffset,
			int outputMemoryHeight, int inputMemoryHeight);
	~MultiplyAllUnitsKernel();

	void InitializeCompilerOptions();
	void SetUseRelaxedMath(bool value);
	void SetUseConstantInput(bool value);
	void SetUseConstantInputDelta(bool value);

	void SetActivationFunction(ATMLActivationFunction activationFunction);

	void SetInput(OpenCLMemory* input);
	void SetInputDelta(OpenCLMemory* inputDelta);
	void SetOutput(OpenCLMemory* output);

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_MULTIPLYALLUNITSKERNEL_H_ */