/*
 * ZeroBorderKenel.h
 *
 *  Created on: May 23, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_ZEROBORDERKENEL_H_
#define ATML_CNNOPENCL_ZEROBORDERKENEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace ATML::Helper;
using namespace std;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class ZeroBorderKenel: public OpenCLKernelProgram
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
	int borderSize;
	int inputStride;
	int inputMemoryHeight;
	int inputUnitOffset;

public:
	ZeroBorderKenel(int dataWidth, int dataHeight, int dataUnits,
			int borderStartLeft, int borderStartRight, int borderStartUp,
			int borderStartDown, int borderSize, int inputStride,
			int inputMemoryHeight, int inputUnitOffset);
	~ZeroBorderKenel();

	void SetInputOutput(OpenCLMemory* input);
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
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_ZEROBORDERKENEL_H_ */
