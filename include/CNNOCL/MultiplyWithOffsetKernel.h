/*
 * MultiplyWithOffsetKernel.h
 *
 *  Created on: May 31, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOCL_MULTIPLYWITHOFFSETKERNEL_H_
#define MATUNA_CNNOCL_MULTIPLYWITHOFFSETKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "CNN/MatunaComputationPrecision.h"
#include "OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class MultiplyWithOffsetKernel: public OCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	bool useConstantInput;
	bool useConstantInputDelta;
	bool useRelaxedMath;

	int globalWidth;
	int globalHeight;
	int globalUnits;
	int dataWidth;
	int dataHeight;
	int inputDeltaStride;
	int inputDeltaMemoryHeight;
	int outputStride;
	int outputMemoryHeight;
	int inputStride;
	int inputWidthOffset;
	int inputHeightOffset;
	int inputDeltaWidthOffset;
	int inputDeltaHeightOffset;
	int inputDeltaUnitOffset;
	int outputWidthoffset;
	int outputHeightOffset;
	int outputUnitOffset;

public:
	MultiplyWithOffsetKernel(int globalWidth, int globalHeight, int globalUnits,
			int dataWidth, int dataHeight, int inputDeltaStride,
			int inputDeltaMemoryHeight, int outputStride,
			int outputMemoryHeight, int inputStride, int inputWidthOffset,
			int inputHeightOffset, int inputDeltaWidthOffset,
			int inputDeltaHeightOffset, int inputDeltaUnitOffset,
			int outputWidthoffset, int outputHeightOffset,
			int outputUnitOffset);
	~MultiplyWithOffsetKernel();

	void SetInput(OCLMemory* input);
	void SetInputDelta(OCLMemory* inputDelta);
	void SetOutput(OCLMemory* output);

	void SetConstantInput(bool value);
	void SetConstantInputDelta(bool value);
	void SetRelaxedMath(bool value);

	void InitializeCompilerOptions();

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOCL_MULTIPLYWITHOFFSETKERNEL_H_ */
