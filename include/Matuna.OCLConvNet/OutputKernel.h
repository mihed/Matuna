/*
 * OutputKernel.h
 *
 *  Created on: May 15, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLConvNet_OUTPUTKERNEL_H_
#define MATUNA_OCLConvNet_OUTPUTKERNEL_H_

#include "Matuna.OCLHelper/OCLKernelProgram.h"
#include "Matuna.OCLHelper/OCLMemory.h"
#include "Matuna.ConvNet/MatunaComputationPrecision.h"
#include "Matuna.ConvNet/MatunaActivationFunctionEnum.h"
#include "Matuna.ConvNet/MatunaErrorFunctionEnum.h"
#include <string>
#include <vector>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class OutputKernel: public OCLKernelProgram
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
	MatunaActivationFunction activationFunction;
	MatunaComputationPrecision computationPrecision;
	MatunaErrorFunction errorFunction;
	int unitsCount;
	int inputOffset;
	int outputOffset;

public:
	OutputKernel(int unitsCount, int inputOffset, int outputOffset);
	~OutputKernel();

	void SetConstantInput(bool value);
	void SetConstantTarget(bool value);
	void SetUseRelaxedMath(bool value);
	void SetActivationFunction(MatunaActivationFunction activationFunction);
	void SetComputationPrecision(MatunaComputationPrecision computationPrecision);
	void SetErrorFunction(MatunaErrorFunction errorFunction);

	//Changed for every execution
	void SetInput(OCLMemory* input);
	//Changed for every execution
	void SetTarget(OCLMemory* target);
	//Changed for every execution
	void SetOutput(OCLMemory* output);

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

#endif /* MATUNA_OCLConvNet_OUTPUTKERNEL_H_ */
