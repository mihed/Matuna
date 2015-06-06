/*
 * PerceptronKernel.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_PERCEPTRONKERNEL_H_
#define MATUNA_CNNOPENCL_PERCEPTRONKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include "CNN/MatunaActivationFunctionEnum.h"
#include "CNN/MatunaComputationPrecision.h"
#include <memory>
#include <tuple>
#include <string>
#include <vector>

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
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

	MatunaActivationFunction activationFunction;
	MatunaComputationPrecision computationPrecision;

public:
	ForwardPerceptronKernel(int inputUnitsCount, int unitsCount);
	~ForwardPerceptronKernel();

	void SetUseConstantWeights(bool value);
	void SetUseConstantInput(bool value);
	void SetUseConstantBiases(bool value);
	void SetUseRelaxedMath(bool value);
	void SetActivationFunction(MatunaActivationFunction activationFunction);
	void SetComputationPrecision(MatunaComputationPrecision computationPrecision);
	void SetWeights(OpenCLMemory* weights);
	void SetBiases(OpenCLMemory* biases);

	//Changed for every execution
	void SetInput(OpenCLMemory* input);
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

}
/* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOPENCL_PERCEPTRONKERNEL_H_ */
