/*
 * BackPerceptronKernel.h
 *
 *  Created on: May 16, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_BACKPERCEPTRONKERNEL_H_
#define ATML_CNNOPENCL_BACKPERCEPTRONKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "CNN/ATMLActivationFunctionEnum.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class BackPerceptronKernel: public OpenCLKernelProgram
{
private:
	OpenCLMemory* weights;
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;
	ATMLActivationFunction activationFunction;
	bool useRelaxedMath;
	bool useConstantInput;
	bool useConstantDeltaInput;
	bool useConstantWeights;
	int inputUnits;
	int outputUnits;
	int inputDeltaOffset;
	int inputOffset;
	int outputOffset;
public:
	BackPerceptronKernel(int inputUnits, int outputUnits, int inputDeltaOffset,
			int inputOffset, int outputOffset);
	~BackPerceptronKernel();

	void SetUseConstantWeights(bool value);
	void SetUseConstantInput(bool value);
	void SetUseConstantDeltaInput(bool value);
	void SetUseRelaxedMath(bool value);
	void SetActivationFunction(ATMLActivationFunction activationFunction);
	void SetWeights(OpenCLMemory* weights);

	//Changed for every execution
	void SetInput(OpenCLMemory* input);
	//Changed for every execution
	void SetDeltaInput(OpenCLMemory* deltaInput);
	//Changed for every execution
	void SetOutput(OpenCLMemory* output);

	void InitializeArguments();
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

#endif /* ATML_CNNOPENCL_BACKPERCEPTRONKERNEL_H_ */
