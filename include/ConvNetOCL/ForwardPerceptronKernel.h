/*
 * PerceptronKernel.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_ConvNetOCL_PERCEPTRONKERNEL_H_
#define MATUNA_ConvNetOCL_PERCEPTRONKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "OCLHelper/OCLMemory.h"
#include "ConvNet/MatunaActivationFunctionEnum.h"
#include "ConvNet/MatunaComputationPrecision.h"
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
class ForwardPerceptronKernel: public OCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;

	OCLMemory* weights;
	OCLMemory* biases;

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
	void SetWeights(OCLMemory* weights);
	void SetBiases(OCLMemory* biases);

	//Changed for every execution
	void SetInput(OCLMemory* input);
	//Changed for every execution
	void SetOutput(OCLMemory* output);

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

#endif /* MATUNA_ConvNetOCL_PERCEPTRONKERNEL_H_ */
