/*
 * SumUnitKernel.h
 *
 *  Created on: May 31, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLConvNet_SUMUNITKERNEL_H_
#define MATUNA_OCLConvNet_SUMUNITKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "ConvNet/MatunaActivationFunctionEnum.h"
#include "ConvNet/MatunaComputationPrecision.h"
#include "OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class SumUnitKernel: public OCLKernelProgram
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

	void SetInput(OCLMemory* input);
	void SetOutput(OCLMemory* output);

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
} /* namespace Matuna */

#endif /* MATUNA_OCLConvNet_SUMUNITKERNEL_H_ */
