/*
 * ImageErrorKernel.h
 *
 *  Created on: Jun 1, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_IMAGEERRORKERNEL_H_
#define MATUNA_CNNOPENCL_IMAGEERRORKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include "CNN/MatunaComputationPrecision.h"
#include "CNN/MatunaErrorFunctionEnum.h"
#include <string>
#include <vector>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class ImageErrorKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;
	bool useRelaxedMath;
	bool useConstantInput;
	bool useConstantTarget;
	MatunaErrorFunction errorFunction;
	MatunaComputationPrecision computationPrecision;

	int dataWidth;
	int dataHeight;
	int dataUnits;
	int inputWidthOffset;
	int inputHeightOffset;
	int inputUnitOffset;
	int inputStride;
	int inputMemoryHeight;
public:
	ImageErrorKernel(int dataWidth, int dataHeight, int dataUnits,
			int inputWidthOffset, int inputHeightOffset, int inputUnitOffset,
			int inputStride, int inputMemoryHeight);
	~ImageErrorKernel();

	void SetConstantInput(bool value);
	void SetConstantTarget(bool value);
	void SetUseRelaxedMath(bool value);
	void SetErrorFunction(MatunaErrorFunction errorFunction);
	void SetComputationPrecision(MatunaComputationPrecision computationPrecision);

	//Changed for every execution
	void SetInput(OpenCLMemory* input);
	//Changed for every execution
	void SetTarget(OpenCLMemory* target);
	//Changed for every execution
	void SetError(OpenCLMemory* error);

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

#endif /* MATUNA_CNNOPENCL_IMAGEERRORKERNEL_H_ */
