/*
 * BackConvolutionKernel.h
 *
 *  Created on: May 24, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_BACKCONVOLUTIONKERNEL_H_
#define MATUNA_CNNOPENCL_BACKCONVOLUTIONKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class BackConvolutionKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	int globalWidth;
	int globalHeight;
	int globalUnits;
	int filterWidth;
	int filterHeight;
	int inputUnitOfffset;
	int inputWidthOffset;
	int inputHeightOffset;
	int outputWidthOffset;
	int outputHeightOffset;
	int inputStride;
	int outputStride;
	int inputMemoryHeight;

	bool useLocalMemory;
	bool useRelaxedMath;
	bool useConstantDeltaInput;
	bool useConstantFilters;
public:
	BackConvolutionKernel(int globalWidth, int globalHeight, int globalUnits,
			int filterWidth, int filterHeight, int inputUnitOfffset,
			int inputWidthOffset, int inputHeightOffset, int outputWidthOffset,
			int outputHeightOffset, int inputStride, int outputStride,
			int inputMemoryHeight, bool useLocalmemory =
					false);
	~BackConvolutionKernel();

	void SetFilters(OpenCLMemory* filters);

	//Changed for every execution
	void SetDeltaInput(OpenCLMemory* deltaInput);
	//Changed for every execution
	void SetOutput(OpenCLMemory* output);

	//Observe that the amount of memory allocated is a multiple of the units
	void SetLocalWorkGroup(int width, int height);

	void InitializeCompilerOptions();

	void SetConstantInputDelta(bool value);
	void SetConstantFilters(bool value);
	void SetRelaxedMath(bool value);

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOPENCL_BACKCONVOLUTIONKERNEL_H_ */
