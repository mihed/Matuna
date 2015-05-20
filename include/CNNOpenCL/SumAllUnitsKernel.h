/*
 * SumAllUnitsKernel.h
 *
 *  Created on: May 20, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_SUMALLUNITSKERNEL_H_
#define ATML_CNNOPENCL_SUMALLUNITSKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace ATML::Helper;
using namespace std;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class SumAllUnitsKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	bool useConstantInput;
	bool useRelaxedMath;

	int globalWidth;
	int globalHeight;
	int unitsToSum;
	int inputWidthOffset;
	int inputHeightOffset;
	int inputUnitOffset;
	int inputStride;
	int inputMemoryHeight;
	int outputWidthOffset;
	int outputHeightOffset;
	int outputStride;
	int outputMemoryHeight;

public:
	SumAllUnitsKernel(int globalWidth, int globalHeight, int unitsToSum,
			int inputWidthOffset, int inputHeightOffset, int inputUnitOffset,
			int inputStride, int inputMemoryHeight, int outputWidthOffset,
			int outputHeightOffset, int outputStride, int outputMemoryHeight);

	~SumAllUnitsKernel();

	void InitializeCompilerOptions();

	void SetUseConstantInput(bool value);
	void SetUseRelaxedMath(bool value);

	void SetInput(OpenCLMemory* input);
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

#endif /* ATML_CNNOPENCL_SUMALLUNITSKERNEL_H_ */
