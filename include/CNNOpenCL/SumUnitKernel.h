/*
 * SumUnitKernel.h
 *
 *  Created on: May 31, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_SUMUNITKERNEL_H_
#define ATML_CNNOPENCL_SUMUNITKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "CNN/ATMLActivationFunctionEnum.h"
#include "CNN/ATMLComputationPrecision.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace ATML::Helper;
using namespace std;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class SumUnitKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	bool useConstantInput;
	bool useRelaxedMath;

	int inputStride;
	int inputMemoryHeight;
	int inputWidthOffset;
	int inputHeightOffset;
	int inputUnitOffset;
	int dataWidth;
	int dataHeight;

	int globalUnits;
	int outputOffset;

public:
	SumUnitKernel(int inputStride, int inputMemoryHeight, int inputWidthOffset,
			int inputHeightOffset, int inputUnitOffset, int outputOffset,
			int globalUnits, int dataWidth, int dataHeight);
	~SumUnitKernel();

	void SetInput(OpenCLMemory* input);
	void SetOutput(OpenCLMemory* output);

	void SetConstantInput(bool value);
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
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_SUMUNITKERNEL_H_ */
