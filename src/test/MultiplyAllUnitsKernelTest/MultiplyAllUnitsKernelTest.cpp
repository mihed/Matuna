/*
 * MultiplyAllUnitsKernelTest.cpp
 *
 *  Created on: May 24, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/MultiplyAllUnitsKernel.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace Matuna::Helper;
using namespace Matuna::Math;
using namespace Matuna::MachineLearning;

float SigmoidActivationDerivativeFloat(float x)
{
	return x * (1 - x);
}

double SigmoidActivationDerivativeDouble(double x)
{
	return  x * (1 - x);
}

float TanhActivationDerivativeFloat(float x)
{
	return 0.6666666f * (1.7159f - (x * x) / 1.7159f);
}

double TanhActivationDerivativeDouble(double x)
{
	return 0.666666666666666 * (1.7159 - (x * x) / 1.7159);
}

SCENARIO("Multiplying all the inputs with a single unit")
{
	random_device tempDevice;
	mt19937 mt(tempDevice());
	uniform_int_distribution<int> dimensionGenerator(1, 1000);
	uniform_int_distribution<int> unitGenerator(1, 40);
	uniform_int_distribution<int> paddingGenerator(1, 20);
	uniform_int_distribution<int> activationGenerator(1, 3);

	WHEN("Multiplying all the inputs with a single unit")
	{
		for (int dummy = 0; dummy < 5; dummy++)
		{
			auto platformInfos = OpenCLHelper::GetPlatformInfos();
			for (auto& platformInfo : platformInfos)
			{
				auto context = OpenCLHelper::GetContext(platformInfo);
				auto devices = context->GetDevices();
				for (auto device : devices)
				{
					vector<OpenCLDevice*> oneDeviceVector;
					oneDeviceVector.push_back(device);

					int width = dimensionGenerator(mt);
					int height = dimensionGenerator(mt);
					int units = unitGenerator(mt);

					int activation = activationGenerator(mt);
					MatunaActivationFunction activationFunction;
					switch (activation)
					{
					case 1:
						activationFunction = MatunaSigmoidActivation;
						break;
					case 2:
						activationFunction = MatunaLinearActivation;
						break;
					case 3:
						activationFunction = MatunaTanhActivation;
						break;
					}

					int inputWidthOffset = paddingGenerator(mt);
					int inputWidthPadding = paddingGenerator(mt);

					int inputHeightOffset = paddingGenerator(mt);
					int inputHeightPadding = paddingGenerator(mt);

					int inputUnitOffset = paddingGenerator(mt);
					int inputUnitPadding = paddingGenerator(mt);

					int outputWidthOffset = paddingGenerator(mt);
					int outputWidthPadding = paddingGenerator(mt);

					int outputHeightOffset = paddingGenerator(mt);
					int outputHeightPadding = paddingGenerator(mt);

					int outputUnitOffset = paddingGenerator(mt);
					int outputUnitPadding = paddingGenerator(mt);

					int inputDeltaWidthOffset = paddingGenerator(mt);
					int inputDeltaWidthPadding = paddingGenerator(mt);

					int inputDeltaHeightOffset = paddingGenerator(mt);
					int inputDeltaHeighPadding = paddingGenerator(mt);

					int totalInputWidth = inputWidthPadding + inputWidthOffset + width;
					int totalInputHeight = inputHeightPadding + inputHeightOffset + height;
					int totalInputUnits = inputUnitOffset + units + inputUnitPadding;
					int totalInputElementsPerUnit = totalInputWidth *  totalInputHeight;

					int totalOutputWidth = outputWidthOffset + outputWidthPadding + width;
					int totalOutputHeight = outputHeightOffset + outputHeightPadding + height;
					int totalOutputUnits = outputUnitOffset + outputUnitPadding + units;
					int totalOutputElementsPerUnit = totalOutputWidth *  totalOutputHeight;

					int totalDeltaWidth = inputDeltaWidthOffset + inputDeltaWidthPadding + width;
					int totalDeltaHeight = inputDeltaHeighPadding + inputDeltaHeightOffset + height;

					auto inputDeltaMatrix = Matrixf::RandomNormal(totalDeltaHeight, totalDeltaWidth);

					vector<Matrixf> inputMatrices;
					for (int i = 0; i < totalInputUnits; i++)
						inputMatrices.push_back(Matrixf::RandomNormal(totalInputHeight, totalInputWidth));

					INFO("Calculating the manual result");
					vector<Matrixf> manualResults;
					for (int i = inputUnitOffset; i < (units + inputUnitOffset); i++)
					{
						auto temp = inputMatrices[i].GetSubMatrix(inputHeightOffset, inputWidthOffset, height, width);
						switch (activationFunction)
						{
						case MatunaSigmoidActivation:
							temp.Transform(&SigmoidActivationDerivativeFloat);
							break;
						case MatunaTanhActivation:
							temp.Transform(&TanhActivationDerivativeFloat);
							break;
						}

						if (activationFunction != MatunaLinearActivation)
							manualResults.push_back(temp %
							inputDeltaMatrix.GetSubMatrix(inputDeltaHeightOffset, inputDeltaWidthOffset, height, width));
						else
							manualResults.push_back(inputDeltaMatrix.GetSubMatrix(inputDeltaHeightOffset, inputDeltaWidthOffset, height, width));
					}

					INFO("Creating the input and output buffers on the host device");
					unique_ptr<float[]> inputMemoryBuffer(new float[totalInputElementsPerUnit *  totalInputUnits]);
					unique_ptr<float[]> outputMemoryBuffer(new float[totalOutputElementsPerUnit *  totalOutputUnits]);

					INFO("Creating the MultiplyAllUnitsKernel");
					MultiplyAllUnitsKernel<float> kernel(width, height, units, totalDeltaWidth,
						totalOutputWidth, totalInputWidth, inputDeltaWidthOffset, inputDeltaHeightOffset,
						outputWidthOffset, outputHeightOffset, outputUnitOffset, inputWidthOffset,
						inputHeightOffset, inputUnitOffset, totalOutputHeight, totalInputHeight);

					kernel.SetActivationFunction(activationFunction);
					kernel.InitializeCompilerOptions();
					context->AddProgramFromSource(&kernel, oneDeviceVector);
					context->AddKernel(&kernel);

					INFO("Creating the OCL buffers");
					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * totalInputElementsPerUnit * totalInputUnits);
					auto inputDeltaMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * totalDeltaHeight * totalDeltaWidth);
					auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(float) *  totalOutputElementsPerUnit *  totalOutputUnits);

					INFO("Writing the input matrices into the input buffers");
					auto rawInputPointer = inputMemoryBuffer.get();
					for (int i = 0; i < totalInputUnits; i++)
					{
						memcpy(rawInputPointer, inputMatrices[i].Data, inputMatrices[i].ElementCount() * sizeof(float));
						rawInputPointer += inputMatrices[i].ElementCount();
					}

					INFO("Writing the host buffers into device memory");
					device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), inputMemoryBuffer.get());
					device->WriteMemory(inputDeltaMemory.get(), inputDeltaMemory->ByteSize(), inputDeltaMatrix.Data);
					device->WaitForDeviceQueue(0);

					INFO("Setting the kernel arguments");
					kernel.SetInput(inputMemory.get());
					kernel.SetOutput(outputMemory.get());
					kernel.SetInputDelta(inputDeltaMemory.get());

					INFO("Executing the kernel");
					device->ExecuteKernel(&kernel);
					device->WaitForDeviceQueue(0);

					INFO("Reading the output memory from the device");
					device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), outputMemoryBuffer.get());
					device->WaitForDeviceQueue(0);

					for (int i = 0; i < units; i++)
					{
						Matrixf outputMemoryMatrix(totalOutputHeight, totalOutputWidth, outputMemoryBuffer.get() + (i + outputUnitOffset) *  totalOutputElementsPerUnit);
						auto temp = outputMemoryMatrix.GetSubMatrix(outputHeightOffset, outputWidthOffset, height, width);
						auto difference = (temp - manualResults[i]).Norm2Square() / temp.ElementCount();
						cout << "difference: " << difference << endl;
						CHECK(difference < 1E-12);
					}
				}
			}
		}
	}
}

