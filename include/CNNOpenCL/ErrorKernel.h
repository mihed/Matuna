/*
 * ErrorKernel.h
 *
 *  Created on: May 15, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_ERRORKERNEL_H_
#define ATML_CNNOPENCL_ERRORKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include "CNN/ATMLComputationPrecision.h"
#include "CNN/ATMLErrorFunctionEnum.h"
#include <string>
#include <vector>

using namespace std;
using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class ErrorKernel: public OpenCLKernelProgram
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
	ATMLErrorFunction errorFunction;
	ATMLComputationPrecision computationPrecision;
	int units; 
	int unitOffset;
public:
	ErrorKernel(int units, int unitOffset);
	~ErrorKernel();

	void SetConstantInput(bool value);
	void SetConstantTarget(bool value);
	void SetUseRelaxedMath(bool value);
	void SetErrorFunction(ATMLErrorFunction errorFunction);
	void SetComputationPrecision(ATMLComputationPrecision computationPrecision);

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
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_ERRORKERNEL_H_ */
