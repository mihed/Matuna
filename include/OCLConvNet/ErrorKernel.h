/*
 * ErrorKernel.h
 *
 *  Created on: May 15, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLConvNet_ERRORKERNEL_H_
#define MATUNA_OCLConvNet_ERRORKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "OCLHelper/OCLMemory.h"
#include "ConvNet/MatunaComputationPrecision.h"
#include "ConvNet/MatunaErrorFunctionEnum.h"
#include <string>
#include <vector>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class ErrorKernel: public OCLKernelProgram
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
	int units; 
	int unitOffset;
public:
	ErrorKernel(int units, int unitOffset);
	~ErrorKernel();

	void SetConstantInput(bool value);
	void SetConstantTarget(bool value);
	void SetUseRelaxedMath(bool value);
	void SetErrorFunction(MatunaErrorFunction errorFunction);
	void SetComputationPrecision(MatunaComputationPrecision computationPrecision);

	//Changed for every execution
	void SetInput(OCLMemory* input);
	//Changed for every execution
	void SetTarget(OCLMemory* target);
	//Changed for every execution
	void SetError(OCLMemory* error);

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

#endif /* MATUNA_OCLConvNet_ERRORKERNEL_H_ */
