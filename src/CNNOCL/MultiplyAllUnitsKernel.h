/*
 * MultiplyAllUnitsKernel.h
 *
 *  Created on: May 24, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOCL_MULTIPLYALLUNITSKERNEL_H_
#define MATUNA_CNNOCL_MULTIPLYALLUNITSKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "CNN/MatunaActivationFunctionEnum.h"
#include "OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class MultiplyAllUnitsKernel: public OCLKernelProgram
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
	MatunaActivationFunction activationFunction;

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

	void SetActivationFunction(MatunaActivationFunction activationFunction);

	void SetInput(OCLMemory* input);
	void SetInputDelta(OCLMemory* inputDelta);
	void SetOutput(OCLMemory* output);

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOCL_MULTIPLYALLUNITSKERNEL_H_ */
