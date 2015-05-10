/*
 * PerceptronKernel.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_PERCEPTRONKERNEL_H_
#define ATML_CNNOPENCL_PERCEPTRONKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include "CNN/ATMLActivationFunctionEnum.h"
#include "CNN/ATMLComputationPrecision.h"
#include <memory>
#include <tuple>
#include <string>
#include <vector>

using namespace ATML::Helper;
using namespace std;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class ForwardPerceptronKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;

	OpenCLMemory* weights;
	OpenCLMemory* biases;

	string kernelName;
	string programName;
	string compilerOptions;

	bool useConstantWeights;
	bool useConstantInput;
	bool useConstantBiases;
	bool useRelaxedMath;

	int inputUnitsCount;
	int unitsCount;

	ATMLActivationFunction activationFunction;
	ATMLComputationPrecision computationPrecision;

public:
	ForwardPerceptronKernel(int inputUnitsCount, int unitsCount);
	~ForwardPerceptronKernel();

	void SetUseConstantWeights(bool value);
	void SetUseConstantInput(bool value);
	void SetUseConstantBiases(bool value);
	void SetUseRelaxedMath(bool value);
	void SetActivationFunction(ATMLActivationFunction activationFunction);
	void SetComputationPrecision(ATMLComputationPrecision computationPrecision);
	void SetWeights(OpenCLMemory* weights);
	void SetBiases(OpenCLMemory* biases);

	//Changed for every execution
	void SetInput(OpenCLMemory* input);
	//Changed for every execution
	void SetOutput(OpenCLMemory* output);

	void InitializeArgumentsAndCompilerOptions();

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

}
/* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_PERCEPTRONKERNEL_H_ */
