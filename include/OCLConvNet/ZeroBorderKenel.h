/*
 * ZeroBorderKenel.h
 *
 *  Created on: May 23, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLConvNet_ZEROBORDERKENEL_H_
#define MATUNA_OCLConvNet_ZEROBORDERKENEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class ZeroBorderKenel: public OCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	bool useRelaxedMath;

	int dataWidth;
	int dataHeight;
	int borderStartLeft;
	int borderStartRight;
	int borderStartUp;
	int borderStartDown;
	int borderHorizontalSize;
	int borderVerticalSize;
	int inputStride;
	int inputMemoryHeight;
	int inputUnitOffset;

public:
	ZeroBorderKenel(int dataWidth, int dataHeight, int dataUnits,
			int borderStartLeft, int borderStartRight, int borderStartUp,
			int borderStartDown, int borderHorizontalSize, int borderVerticalSize, int inputStride,
			int inputMemoryHeight, int inputUnitOffset);
	~ZeroBorderKenel();

	void SetInputOutput(OCLMemory* input);
	void InitializeCompilerOptions();
	void SetUseRelaxedMath(bool value);

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_OCLConvNet_ZEROBORDERKENEL_H_ */
