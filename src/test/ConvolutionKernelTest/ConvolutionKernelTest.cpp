/*
 * ConvolutionKernelTest.cpp
 *
 *  Created on: May 21, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "OpenCLHelper/OpenCLKernelInfo.h"
#include "CNNOpenCL/ConvolutionKernel.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace ATML::Helper;
using namespace ATML::Math;
using namespace ATML::MachineLearning;

float SigmoidActivationFloat(float x)
{
	return 1 / (1 + exp(-x));
}

double SigmoidActivationDouble(double x)
{
	return 1 / (1 + exp(-x));
}

float TanhActivationFloat(float x)
{
	return 1.7159f * tanh(0.6666666f * x);
}

double TanhActivationDouble(double x)
{
	return 1.7159 * tanh(0.666666666666666 * x);
}

SCENARIO("Performing convolution on a single input with multiple filters")
{

	WHEN("Convolving with local memory, offset and padding")
	{

		for (int dummy = 0; dummy < 10; dummy++)
		{
			auto platformInfos = OpenCLHelper::GetPlatformInfos();
			for (auto& platformInfo : platformInfos)
			{
				auto context = OpenCLHelper::GetContext(platformInfo);
				auto devices = context->GetDevices();
				for (auto device : devices)
				{
					auto maximumLocalMemory = device->DeviceInfo().LocalMemorySize();
					auto maximumLocalWorkGroupDimensions = device->DeviceInfo().MaxWorkItemSizes();

					INFO("Starting by examinating the local memory properties of the device");
					random_device tempDevice;
					mt19937 mt(tempDevice());
					uniform_int_distribution<int> localDimensionGenerator(1, 32); //To this, we need to add the filter width / height
					uniform_int_distribution<int> imageGenerator(1, 30);
					uniform_int_distribution<int> filterDimensionGenerator(1, 20);
					uniform_int_distribution<int> activationGenerator(1, 3);
					uniform_int_distribution<int> paddingGenerator(0, 50);
					normal_distribution<float> distribution;

					int filterUnits = filterDimensionGenerator(mt);
					int filterWidth = filterDimensionGenerator(mt);
					int filterHeight = filterDimensionGenerator(mt);
					int localWidth = localDimensionGenerator(mt);
					int localHeight = localDimensionGenerator(mt);
					int outputWidth = localWidth * imageGenerator(mt);
					int outputHeight = localHeight * imageGenerator(mt);
					int imageWidth = outputWidth + filterWidth - 1;
					int imageHeight = outputHeight + filterHeight - 1;

					int inputWidthOffset = paddingGenerator(mt);
					int inputHeightOffset = paddingGenerator(mt);
					int outputWidthOffset = paddingGenerator(mt);
					int outputHeightOffset = paddingGenerator(mt);
					int outputUnitOffset = paddingGenerator(mt);

					int inputWidthPadding = paddingGenerator(mt);
					int inputHeightPadding = paddingGenerator(mt);
					int outputWidthPadding = paddingGenerator(mt);
					int outputHeightPadding = paddingGenerator(mt);
					int outputUnitPadding = paddingGenerator(mt);

					vector<OpenCLDevice*> oneDeviceVector;
					oneDeviceVector.push_back(device);

					auto activation = activationGenerator(mt);
					ATMLActivationFunction activationFunction;
					switch (activation)
					{
					case 1:
						activationFunction = ATMLSigmoidActivation;
						break;
					case 2:
						activationFunction = ATMLLinearActivation;
						break;
					case 3:
						activationFunction = ATMLTanhActivation;
						break;
					}

					Matrix<float> input = Matrix<float>::RandomNormal(imageHeight + inputHeightOffset + inputHeightPadding, imageWidth + inputWidthOffset + inputWidthPadding);
					vector<Matrix<float>> filters;
					vector<float> biases;

					INFO("Constructing random biases");
					for (int i = 0; i < filterUnits; i++)
						biases.push_back(distribution(mt));

					INFO("Constructing random filters");
					for (int i = 0; i < filterUnits; i++)
						filters.push_back(Matrix<float>::RandomNormal(filterHeight, filterWidth));

					INFO("Putting the filters into contiguous memory");
					unique_ptr<float[]> filterMemoryBuffer(new float[filterWidth * filterHeight * filterUnits]);
					auto rawPointer = filterMemoryBuffer.get();
					for (int i = 0; i < filterUnits; i++)
					{
						auto& unit = filters[i];
						memcpy(rawPointer, unit.Data, unit.ElementCount() * sizeof(float));
						rawPointer += unit.ElementCount();
					}

					INFO("Putting the biases into contigous memory");
					unique_ptr<float[]> biasMemoryBuffer(new float[filterUnits]);
					for (int i = 0; i < filterUnits; i++)
						biasMemoryBuffer[i] = biases[i];

					INFO("Initializing the convolution kernel");
					ConvolutionKernel<float> convolutionKernel(filterUnits, outputWidth, outputHeight,
						filterWidth, filterHeight, inputWidthOffset, inputHeightOffset, outputWidthOffset, outputHeightOffset, outputUnitOffset,
						(outputWidth + outputWidthPadding + outputWidthOffset), (imageWidth + inputWidthOffset + inputWidthPadding),
						(outputWidth + outputWidthPadding + outputWidthOffset) * (outputHeight + outputHeightPadding + outputHeightOffset),
						filterWidth * filterHeight, true);

					convolutionKernel.SetActivationFunction(activationFunction);
					convolutionKernel.InitializeCompilerOptions();

					context->AddProgramFromSource(&convolutionKernel, oneDeviceVector);
					context->AddKernel(&convolutionKernel);

					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * (imageWidth + inputWidthOffset + inputWidthPadding) * (imageHeight + inputHeightOffset + inputHeightPadding));
					device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input.Data);

					auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(float) * (outputWidth + outputWidthPadding + outputWidthOffset) * (outputHeight + outputHeightPadding + outputHeightOffset)
						* (filterUnits + outputUnitOffset + outputUnitPadding));

					auto filterMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * filterWidth * filterHeight * filterUnits);
					device->WriteMemory(filterMemory.get(), filterMemory->ByteSize(), filterMemoryBuffer.get());
					auto biasMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * filterUnits);
					device->WriteMemory(biasMemory.get(), biasMemory->ByteSize(), biasMemoryBuffer.get());

					convolutionKernel.SetInput(inputMemory.get());
					convolutionKernel.SetBiases(biasMemory.get());
					convolutionKernel.SetFilters(filterMemory.get());
					convolutionKernel.SetOutput(outputMemory.get());

					//Ok, here we need to query the kernel info
					OpenCLKernelInfo kernelInfo = device->GetKernelInfo(&convolutionKernel);

					//Check that the local size is valid (I don't care about memory overflow here)
					auto maxWorkGroupSize = kernelInfo.KernelWorkGroupSize();
					while ((localWidth * localHeight) > maxWorkGroupSize || (outputWidth % localWidth != 0 || outputHeight % localHeight != 0))
					{
						localWidth--;
						localHeight--;
						if (outputWidth % localWidth != 0)
							continue;
						if (outputHeight % localHeight != 0)
							continue;
					}

					if (maximumLocalMemory < ((localWidth + filterWidth - 1)* (localHeight + filterHeight - 1) * sizeof(float)))
						WARN("We are using too much local memory. This will probably slow down the execution.");

					if (localHeight < 1)
					{
						WARN("No fitting local height, skipping");
						continue;
					}

					if (localWidth < 1)
					{
						WARN("No fitting local height, skipping");
						continue;
					}

					auto temp = outputWidth % localWidth;
					CHECK(temp == 0);
					temp = outputHeight % localHeight;
					CHECK(temp == 0);

					//Let us now create set the local size
					convolutionKernel.SetLocalWorkGroup(localWidth, localHeight);

					device->ExecuteKernel(&convolutionKernel);
					device->WaitForDeviceQueue(0);
					INFO("Creating the output memory");
					unique_ptr<float[]> outputMemoryBuffer(new float[(outputWidth + outputWidthPadding + outputWidthOffset) * (outputHeight + outputHeightPadding + outputHeightOffset)
						* (filterUnits + outputUnitOffset + outputUnitPadding)]);
					device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), outputMemoryBuffer.get());
					device->WaitForDeviceQueue(0);

					INFO("Calculating the manual result");
					vector<Matrix<float>> manualResults;
					for (int i = 0; i < filterUnits; i++)
					{
						//cout << "Input: \n" << input.GetString() << endl;
						//cout << "Filter: \n" << filters[i].GetString() << endl;
						//cout << "Bias: \n" << biases[i] << endl;
						auto manualResult = input.GetSubMatrix(inputHeightOffset, inputWidthOffset, imageHeight, imageWidth).Convolve(filters[i]) + biases[i];
						switch (activationFunction)
						{
						case ATMLSigmoidActivation:
							manualResult.Transform(&SigmoidActivationFloat);
							break;
						case ATMLTanhActivation:
							manualResult.Transform(&TanhActivationFloat);
							break;
						}

						manualResults.push_back(manualResult);
					}

					INFO("Comparing the manual result to the kernel result");
					auto rawTemp = outputMemoryBuffer.get();

					rawTemp = rawTemp + (outputHeight + outputHeightPadding + outputHeightOffset) * (outputWidth + outputWidthPadding + outputWidthOffset) * outputUnitOffset;

					for (int i = 0; i < filterUnits; i++)
					{
						Matrix<float> tempKernelResult((outputHeight + outputHeightPadding + outputHeightOffset), (outputWidth + outputWidthPadding + outputWidthOffset), rawTemp);
						rawTemp += tempKernelResult.ElementCount();
						Matrix<float> kernelResult = tempKernelResult.GetSubMatrix(outputHeightOffset, outputWidthOffset, outputHeight, outputWidth);
						auto difference = (kernelResult - manualResults[i]).Norm2Square() / kernelResult.ElementCount();
						cout << "difference: " << difference << endl;
						CHECK(difference < 1E-7);
					}

				}
			}
		}
	}


	WHEN("Convolving with padding and offset, without local memory")
	{
		random_device tempDevice;
		mt19937 mt(tempDevice());
		uniform_int_distribution<int> dimensionGenerator(30, 1000);
		uniform_int_distribution<int> paddingGenerator(0, 50);
		uniform_int_distribution<int> filterDimensionGenerator(1, 30);
		uniform_int_distribution<int> activationGenerator(1, 3);
		normal_distribution<float> distribution;

		for (int dummy = 0; dummy < 10; dummy++)
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

					int filterWidth = filterDimensionGenerator(mt);
					int filterHeight = filterDimensionGenerator(mt);
					int filterUnits = filterDimensionGenerator(mt);

					int imageWidth = dimensionGenerator(mt);
					int imageHeight = dimensionGenerator(mt);

					int outputWidth = imageWidth - filterWidth + 1;
					int outputHeight = imageHeight - filterHeight + 1;

					int inputWidthOffset = paddingGenerator(mt);
					int inputHeightOffset = paddingGenerator(mt);
					int outputWidthOffset = paddingGenerator(mt);
					int outputHeightOffset = paddingGenerator(mt);
					int outputUnitOffset = paddingGenerator(mt);

					int inputWidthPadding = paddingGenerator(mt);
					int inputHeightPadding = paddingGenerator(mt);
					int outputWidthPadding = paddingGenerator(mt);
					int outputHeightPadding = paddingGenerator(mt);
					int outputUnitPadding = paddingGenerator(mt);

					Matrix<float> input = Matrix<float>::RandomNormal(imageHeight + inputHeightOffset + inputHeightPadding, imageWidth + inputWidthOffset + inputWidthPadding);
					vector<Matrix<float>> filters;
					vector<float> biases;

					INFO("Constructing random filters");
					for (int i = 0; i < filterUnits; i++)
						filters.push_back(Matrix<float>::RandomNormal(filterHeight, filterWidth));

					auto activation = activationGenerator(mt);
					ATMLActivationFunction activationFunction;
					switch (activation)
					{
					case 1:
						activationFunction = ATMLSigmoidActivation;
						break;
					case 2:
						activationFunction = ATMLLinearActivation;
						break;
					case 3:
						activationFunction = ATMLTanhActivation;
						break;
					}

					INFO("Constructing random biases");
					for (int i = 0; i < filterUnits; i++)
						biases.push_back(distribution(mt));

					INFO("Putting the filters into contiguous memory");
					unique_ptr<float[]> filterMemoryBuffer(new float[filterWidth * filterHeight * filterUnits]);
					auto rawPointer = filterMemoryBuffer.get();
					for (int i = 0; i < filterUnits; i++)
					{
						auto& unit = filters[i];
						memcpy(rawPointer, unit.Data, unit.ElementCount() * sizeof(float));
						rawPointer += unit.ElementCount();
					}

					INFO("Putting the biases into contigous memory");
					unique_ptr<float[]> biasMemoryBuffer(new float[filterUnits]);
					for (int i = 0; i < filterUnits; i++)
						biasMemoryBuffer[i] = biases[i];

					INFO("Creating the output memory");
					unique_ptr<float[]> outputMemoryBuffer(new float[(outputWidth + outputWidthOffset + outputWidthPadding)*
						(outputHeight + outputHeightOffset + outputHeightPadding)*
						(filterUnits + outputUnitOffset + outputUnitPadding)]);

					INFO("Calculating the manual result");
					vector<Matrix<float>> manualResults;
					for (int i = 0; i < filterUnits; i++)
					{
						//cout << "Input: \n" << input.GetString() << endl;
						//cout << "Filter: \n" << filters[i].GetString() << endl;
						//cout << "Bias: \n" << biases[i] << endl;
						auto manualResult = input.GetSubMatrix(inputHeightOffset, inputWidthOffset, imageHeight, imageWidth).Convolve(filters[i]) + biases[i];
						switch (activationFunction)
						{
						case ATMLSigmoidActivation:
							manualResult.Transform(&SigmoidActivationFloat);
							break;
						case ATMLTanhActivation:
							manualResult.Transform(&TanhActivationFloat);
							break;
						}

						manualResults.push_back(manualResult);
					}

					INFO("Initializing the convolution kernel");
					ConvolutionKernel<float> convolutionKernel(filterUnits, outputWidth, outputHeight,
						filterWidth, filterHeight, inputWidthOffset, inputHeightOffset, outputWidthOffset, outputHeightOffset, outputUnitOffset,
						(outputWidth + outputWidthPadding + outputWidthOffset), (imageWidth + inputWidthOffset + inputWidthPadding), 
						(outputWidth + outputWidthPadding + outputWidthOffset) * (outputHeight + outputHeightPadding + outputHeightOffset),
						filterWidth * filterHeight);

					convolutionKernel.SetActivationFunction(activationFunction);
					convolutionKernel.InitializeCompilerOptions();
					context->AddProgramFromSource(&convolutionKernel, oneDeviceVector);
					context->AddKernel(&convolutionKernel);

					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * (imageWidth + inputWidthOffset + inputWidthPadding) * (imageHeight + inputHeightOffset + inputHeightPadding));
					device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input.Data);

					auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(float) * (outputWidth + outputWidthPadding + outputWidthOffset) * (outputHeight + outputHeightPadding + outputHeightOffset) 
						* (filterUnits + outputUnitOffset + outputUnitPadding));

					auto filterMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * filterWidth * filterHeight * filterUnits);
					device->WriteMemory(filterMemory.get(), filterMemory->ByteSize(), filterMemoryBuffer.get());
					auto biasMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * filterUnits);
					device->WriteMemory(biasMemory.get(), biasMemory->ByteSize(), biasMemoryBuffer.get());

					convolutionKernel.SetInput(inputMemory.get());
					convolutionKernel.SetBiases(biasMemory.get());
					convolutionKernel.SetFilters(filterMemory.get());
					convolutionKernel.SetOutput(outputMemory.get());

					INFO("Executing the kernel");
					device->ExecuteKernel(&convolutionKernel);
					device->WaitForDeviceQueue(0);
					device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), outputMemoryBuffer.get());
					device->WaitForDeviceQueue(0);

					INFO("Comparing the manual result to the kernel result");
					auto rawTemp = outputMemoryBuffer.get();

					//We need to increase the raw pointer to account fo the offset
					rawTemp = rawTemp + (outputHeight + outputHeightPadding + outputHeightOffset) * (outputWidth + outputWidthPadding + outputWidthOffset) * outputUnitOffset;

					for (int i = 0; i < filterUnits; i++)
					{
						Matrix<float> tempKernelResult((outputHeight + outputHeightPadding + outputHeightOffset), (outputWidth + outputWidthPadding + outputWidthOffset), rawTemp);
						rawTemp += tempKernelResult.ElementCount();
						Matrix<float> kernelResult = tempKernelResult.GetSubMatrix(outputHeightOffset, outputWidthOffset, outputHeight, outputWidth);
						auto difference = (kernelResult - manualResults[i]).Norm2Square() / kernelResult.ElementCount();
						cout << "difference: " << difference << endl;
						CHECK(difference < 1E-7);
					}
				}
			}
		}
	}


	WHEN("Convolving with local memory and without offset and padding")
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
					auto maximumLocalMemory = device->DeviceInfo().LocalMemorySize();
					auto maximumLocalWorkGroupDimensions = device->DeviceInfo().MaxWorkItemSizes();

					INFO("Starting by examinating the local memory properties of the device");
					random_device tempDevice;
					mt19937 mt(tempDevice());
					uniform_int_distribution<int> localDimensionGenerator(1, 32); //To this, we need to add the filter width / height
					uniform_int_distribution<int> imageGenerator(1, 30);
					uniform_int_distribution<int> filterDimensionGenerator(1, 20);
					uniform_int_distribution<int> activationGenerator(1, 3);
					normal_distribution<float> distribution;

					int filterUnits = filterDimensionGenerator(mt);
					int filterWidth = filterDimensionGenerator(mt);
					int filterHeight = filterDimensionGenerator(mt);
					int localWidth = localDimensionGenerator(mt);
					int localHeight = localDimensionGenerator(mt);
					int outputWidth = localWidth * imageGenerator(mt);
					int outputHeight = localHeight * imageGenerator(mt);
					int imageWidth = outputWidth + filterWidth - 1;
					int imageHeight = outputHeight + filterHeight - 1;

					vector<OpenCLDevice*> oneDeviceVector;
					oneDeviceVector.push_back(device);

					auto activation = activationGenerator(mt);
					ATMLActivationFunction activationFunction;
					switch (activation)
					{
					case 1:
						activationFunction = ATMLSigmoidActivation;
						break;
					case 2:
						activationFunction = ATMLLinearActivation;
						break;
					case 3:
						activationFunction = ATMLTanhActivation;
						break;
					}

					Matrix<float> input = Matrix<float>::RandomNormal(imageHeight, imageWidth);
					vector<Matrix<float>> filters;
					vector<float> biases;

					INFO("Constructing random biases");
					for (int i = 0; i < filterUnits; i++)
						biases.push_back(distribution(mt));

					INFO("Constructing random filters");
					for (int i = 0; i < filterUnits; i++)
						filters.push_back(Matrix<float>::RandomNormal(filterHeight, filterWidth));

					INFO("Putting the filters into contiguous memory");
					unique_ptr<float[]> filterMemoryBuffer(new float[filterWidth * filterHeight * filterUnits]);
					auto rawPointer = filterMemoryBuffer.get();
					for (int i = 0; i < filterUnits; i++)
					{
						auto& unit = filters[i];
						memcpy(rawPointer, unit.Data, unit.ElementCount() * sizeof(float));
						rawPointer += unit.ElementCount();
					}

					INFO("Putting the biases into contigous memory");
					unique_ptr<float[]> biasMemoryBuffer(new float[filterUnits]);
					for (int i = 0; i < filterUnits; i++)
						biasMemoryBuffer[i] = biases[i];

					INFO("Initializing the convolution kernel");
					ConvolutionKernel<float> convolutionKernel(filterUnits, outputWidth, outputHeight,
						filterWidth, filterHeight, 0, 0, 0, 0, 0,
						outputWidth, imageWidth, outputWidth * outputHeight,
						filterWidth * filterHeight, true);

					convolutionKernel.SetActivationFunction(activationFunction);
					convolutionKernel.InitializeCompilerOptions();

					context->AddProgramFromSource(&convolutionKernel, oneDeviceVector);
					context->AddKernel(&convolutionKernel);

					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * imageWidth * imageHeight);
					device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input.Data);

					auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(float) * outputWidth * outputHeight * filterUnits);

					auto filterMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * filterWidth * filterHeight * filterUnits);
					device->WriteMemory(filterMemory.get(), filterMemory->ByteSize(), filterMemoryBuffer.get());
					auto biasMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * filterUnits);
					device->WriteMemory(biasMemory.get(), biasMemory->ByteSize(), biasMemoryBuffer.get());

					convolutionKernel.SetInput(inputMemory.get());
					convolutionKernel.SetBiases(biasMemory.get());
					convolutionKernel.SetFilters(filterMemory.get());
					convolutionKernel.SetOutput(outputMemory.get());

					//Ok, here we need to query the kernel info
					OpenCLKernelInfo kernelInfo = device->GetKernelInfo(&convolutionKernel);

					//Check that the local size is valid (I don't care about memory overflow here)
					auto maxWorkGroupSize = kernelInfo.KernelWorkGroupSize();
					while ((localWidth * localHeight) > maxWorkGroupSize || (outputWidth % localWidth != 0 || outputHeight % localHeight != 0))
					{
						localWidth--;
						localHeight--;
						if (outputWidth % localWidth != 0)
							continue;
						if (outputHeight % localHeight != 0)
							continue;
					}

					if (maximumLocalMemory < ((localWidth + filterWidth - 1)* (localHeight + filterHeight - 1) * sizeof(float)))
						WARN("We are using too much local memory. This will probably slow down the execution.");

					if (localHeight < 1)
					{
						WARN("No fitting local height, skipping");
						continue;
					}

					if (localWidth < 1)
					{
						WARN("No fitting local height, skipping");
						continue;
					}

					auto temp = outputWidth % localWidth;
					CHECK(temp == 0);
					temp = outputHeight % localHeight;
					CHECK(temp == 0);

					//Let us now create set the local size
					convolutionKernel.SetLocalWorkGroup(localWidth, localHeight);

					device->ExecuteKernel(&convolutionKernel);
					device->WaitForDeviceQueue(0);
					INFO("Creating the output memory");
					unique_ptr<float[]> outputMemoryBuffer(new float[outputWidth * outputHeight * filterUnits]);
					device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), outputMemoryBuffer.get());
					device->WaitForDeviceQueue(0);

					INFO("Calculating the manual result");
					vector<Matrix<float>> manualResults;
					for (int i = 0; i < filterUnits; i++)
					{
						//cout << "Input: \n" << input.GetString() << endl;
						//cout << "Filter: \n" << filters[i].GetString() << endl;
						//cout << "Bias: \n" << biases[i] << endl;
						auto manualResult = input.Convolve(filters[i]) + biases[i];
						switch (activationFunction)
						{
						case ATMLSigmoidActivation:
							manualResult.Transform(&SigmoidActivationFloat);
							break;
						case ATMLTanhActivation:
							manualResult.Transform(&TanhActivationFloat);
							break;
						}

						manualResults.push_back(manualResult);
					}

					INFO("Comparing the manual result to the kernel result");
					auto rawTemp = outputMemoryBuffer.get();
					for (int i = 0; i < filterUnits; i++)
					{
						Matrix<float> kernelResult(outputHeight, outputWidth, rawTemp);
						//cout << "Kernel result: \n" << kernelResult.GetString() << endl;
						rawTemp += kernelResult.ElementCount();
						auto difference = (kernelResult - manualResults[i]).Norm2Square() / kernelResult.ElementCount();
						cout << "difference: " << difference << endl;
						CHECK(difference < 1E-7);
					}

				}
			}
		}
	}

	WHEN("Convolving without padding, offset and local memory")
	{
		random_device tempDevice;
		mt19937 mt(tempDevice());
		uniform_int_distribution<int> dimensionGenerator(30, 1000);
		uniform_int_distribution<int> filterDimensionGenerator(1, 30);
		uniform_int_distribution<int> activationGenerator(1, 3);
		normal_distribution<float> distribution;

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

					int filterWidth = filterDimensionGenerator(mt);
					int filterHeight = filterDimensionGenerator(mt);
					int filterUnits = filterDimensionGenerator(mt);

					int imageWidth = dimensionGenerator(mt);
					int imageHeight = dimensionGenerator(mt);

					int outputWidth = imageWidth - filterWidth + 1;
					int outputHeight = imageHeight - filterHeight + 1;

					Matrix<float> input = Matrix<float>::RandomNormal(imageHeight, imageWidth);
					vector<Matrix<float>> filters;
					vector<float> biases;

					INFO("Constructing random filters");
					for (int i = 0; i < filterUnits; i++)
						filters.push_back(Matrix<float>::RandomNormal(filterHeight, filterWidth));

					auto activation = activationGenerator(mt);
					ATMLActivationFunction activationFunction;
					switch (activation)
					{
					case 1:
						activationFunction = ATMLSigmoidActivation;
						break;
					case 2:
						activationFunction = ATMLLinearActivation;
						break;
					case 3:
						activationFunction = ATMLTanhActivation;
						break;
					}

					INFO("Constructing random biases");
					for (int i = 0; i < filterUnits; i++)
						biases.push_back(distribution(mt));

					INFO("Putting the filters into contiguous memory");
					unique_ptr<float[]> filterMemoryBuffer(new float[filterWidth * filterHeight * filterUnits]);
					auto rawPointer = filterMemoryBuffer.get();
					for (int i = 0; i < filterUnits; i++)
					{
						auto& unit = filters[i];
						memcpy(rawPointer, unit.Data, unit.ElementCount() * sizeof(float));
						rawPointer += unit.ElementCount();
					}

					INFO("Putting the biases into contigous memory");
					unique_ptr<float[]> biasMemoryBuffer(new float[filterUnits]);
					for (int i = 0; i < filterUnits; i++)
						biasMemoryBuffer[i] = biases[i];

					INFO("Creating the output memory");
					unique_ptr<float[]> outputMemoryBuffer(new float[outputWidth * outputHeight * filterUnits]);

					INFO("Calculating the manual result");
					vector<Matrix<float>> manualResults;
					for (int i = 0; i < filterUnits; i++)
					{
						//cout << "Input: \n" << input.GetString() << endl;
						//cout << "Filter: \n" << filters[i].GetString() << endl;
						//cout << "Bias: \n" << biases[i] << endl;
						auto manualResult = input.Convolve(filters[i]) + biases[i];
						switch (activationFunction)
						{
						case ATMLSigmoidActivation:
							manualResult.Transform(&SigmoidActivationFloat);
							break;
						case ATMLTanhActivation:
							manualResult.Transform(&TanhActivationFloat);
							break;
						}

						manualResults.push_back(manualResult);
					}

					INFO("Initializing the convolution kernel");
					ConvolutionKernel<float> convolutionKernel(filterUnits, outputWidth, outputHeight,
						filterWidth, filterHeight, 0, 0, 0, 0, 0,
						outputWidth, imageWidth, outputWidth * outputHeight,
						filterWidth * filterHeight);

					convolutionKernel.SetActivationFunction(activationFunction);
					convolutionKernel.InitializeCompilerOptions();
					context->AddProgramFromSource(&convolutionKernel, oneDeviceVector);
					context->AddKernel(&convolutionKernel);

					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * imageWidth * imageHeight);
					device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input.Data);

					auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(float) * outputWidth * outputHeight * filterUnits);

					auto filterMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * filterWidth * filterHeight * filterUnits);
					device->WriteMemory(filterMemory.get(), filterMemory->ByteSize(), filterMemoryBuffer.get());
					auto biasMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) * filterUnits);
					device->WriteMemory(biasMemory.get(), biasMemory->ByteSize(), biasMemoryBuffer.get());

					convolutionKernel.SetInput(inputMemory.get());
					convolutionKernel.SetBiases(biasMemory.get());
					convolutionKernel.SetFilters(filterMemory.get());
					convolutionKernel.SetOutput(outputMemory.get());

					INFO("Executing the kernel");
					device->ExecuteKernel(&convolutionKernel);
					device->WaitForDeviceQueue(0);
					device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), outputMemoryBuffer.get());
					device->WaitForDeviceQueue(0);

					INFO("Comparing the manual result to the kernel result");
					auto rawTemp = outputMemoryBuffer.get();
					for (int i = 0; i < manualResults.size(); i++)
					{
						Matrix<float> kernelResult(outputHeight, outputWidth, rawTemp);
						//cout << "Kernel result: \n" << kernelResult.GetString() << endl;
						//cout << "Manual result: \n" << manualResults[i].GetString() << endl;
						rawTemp += kernelResult.ElementCount();
						auto difference = (kernelResult - manualResults[i]).Norm2Square() / kernelResult.ElementCount();
						cout << "difference: " << difference << endl;
						CHECK(difference < 1E-7);
					}
				}
			}
		}
	}
}
