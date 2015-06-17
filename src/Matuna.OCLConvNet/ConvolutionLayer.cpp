/*
* ConvolutionLayer.cpp
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#include "ConvolutionLayer.h"
#include "Matuna.OCLHelper/OCLProgram.h"
#include "Matuna.Helper/Path.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Converter.h"
#include "CheckPrecision.h"
#include <stdexcept>
#include <type_traits>
#include <random>
#include <stdlib.h>
#include <string.h>

namespace Matuna
{
	namespace MachineLearning
	{
		template<class T>
		ConvolutionLayer<T>::ConvolutionLayer(shared_ptr<OCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const ConvolutionLayerConfig* config) :
		OCLForwardBackPropLayer<T>(context, inputLayerDescriptions,
			backPropActivation, config), convolutionConfig(*config)
		{

			if (inputLayerDescriptions.size() == 0)
				throw invalid_argument(
				"There's no input data descriptions for the convolution layer.");

			if (inputLayerDescriptions.size() != 1)
				throw runtime_error("Not implemented exception");

			if (config->ConnectionType() != MatunaFullConnection)
				throw runtime_error("Not implemented exception");

			//Make sure the type we want to execute is supported on the device.
			static_assert(is_same<cl_double, T>::value || is_same<cl_float, T>::value, "The type is not yet supported");
			for (auto device : context->GetDevices()) 
			{
				auto deviceInfo = device->DeviceInfo();
				CheckPrecision<is_same<cl_double, T>::value>::Check(deviceInfo);
			}

			InitializeMemoryDescriptions(inputLayerDescriptions, config);
		}

		template<class T>
		ConvolutionLayer<T>::~ConvolutionLayer()
		{

		}


		template<class T>
		void ConvolutionLayer<T>::InitializeMemoryDescriptions(const vector<LayerDataDescription>& inputLayerDescriptions, const ConvolutionLayerConfig* config)
		{
			//FIXME: Since we are not using any optimization at the moment, such as local memory.
			//We will not require any padding on the input and output proposals
			for (auto& layerDescription : inputLayerDescriptions)
			{
				LayerMemoryDescription inForwardMemProp;
				inForwardMemProp.Height = layerDescription.Height;
				inForwardMemProp.Width = layerDescription.Width;
				inForwardMemProp.Units = layerDescription.Units;
				inForwardMemProp.HeightOffset = 0;
				inForwardMemProp.UnitOffset = 0;
				inForwardMemProp.WidthOffset = 0;

				this->inForwardPropMemoryProposals.push_back(inForwardMemProp);
				this->outBackPropMemoryProposals.push_back(inForwardMemProp);

				LayerDataDescription outForwardDataDesc;
				outForwardDataDesc.Height = layerDescription.Height
					- config->FilterHeight() + 1;
				outForwardDataDesc.Width = layerDescription.Width
					- config->FilterWidth() + 1;
				outForwardDataDesc.Units = config->FilterCount();
				this->outForwardPropDataDescriptions.push_back(outForwardDataDesc);

				LayerMemoryDescription outForwardMemProp;
				outForwardMemProp.Height = layerDescription.Height
					- config->FilterHeight() + 1;
				outForwardMemProp.Width = layerDescription.Width - config->FilterWidth()
					+ 1;
				outForwardMemProp.Units = config->FilterCount();
				outForwardMemProp.HeightOffset = 0;
				outForwardMemProp.UnitOffset = 0;
				outForwardMemProp.WidthOffset = 0;

				this->outForwardPropMemoryProposals.push_back(outForwardMemProp);

				//Since we will add padding to the input, we will require that we have a border of size filterdimension - 1
				LayerMemoryDescription inBackMemProp;
				inBackMemProp.Height = outForwardDataDesc.Height
					+ 2 * (config->FilterHeight() - 1);
				inBackMemProp.Width = outForwardDataDesc.Width
					+ 2 * (config->FilterWidth() - 1);
				inBackMemProp.Units = config->FilterCount();
				inBackMemProp.HeightOffset = config->FilterHeight() - 1;
				inBackMemProp.WidthOffset = config->FilterWidth() - 1;
				inBackMemProp.UnitOffset = 0;

				this->inBackPropMemoryProposals.push_back(inBackMemProp);
			}

			this->inBackPropDataDescriptions = this->outForwardPropDataDescriptions;
		}

		template<class T>
		ConvolutionLayerConfig ConvolutionLayer<T>::GetConfig() const
		{
			return convolutionConfig;
		}

		template<class T>
		vector<Matrix<T>> ConvolutionLayer<T>::GetFilters() const
		{
			OCLDevice* device = this->context->GetDevices()[0];
			int temp = convolutionConfig.FilterHeight()
				* convolutionConfig.FilterWidth();
			int elementCount = temp * convolutionConfig.FilterCount();
			unique_ptr<T[]> contiguousMemory(new T[elementCount]);

			device->ReadMemory(filters.get(), filters->ByteSize(),
				contiguousMemory.get());
			device->WaitForDeviceQueue(0);

			vector<Matrix<T>> result;
			for (int i = 0; i < convolutionConfig.FilterCount(); i++)
			{
				Matrix<T> filter(convolutionConfig.FilterHeight(),
					convolutionConfig.FilterWidth());
				memcpy(filter.Data, contiguousMemory.get() + i * temp,
					temp * sizeof(T));
				result.push_back(filter);
			}

			return result;
		}

		template<class T>
		vector<T> ConvolutionLayer<T>::GetBiases() const
		{
			OCLDevice* device = this->context->GetDevices()[0];
			vector<T> result;
			result.resize(convolutionConfig.FilterCount());
			device->ReadMemory(biases.get(), biases->ByteSize(), result.data());
			device->WaitForDeviceQueue(0);
			return result;
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeParameters()
		{

			//TODO: There are optimization to be made here if the network is read-only. (i.e. non trainable)
			auto filterElementCount = convolutionConfig.FilterCount()
				* convolutionConfig.FilterWidth()
				* convolutionConfig.FilterHeight();
			auto biasElementCount = convolutionConfig.FilterCount();
			filters = move(
				this->context->CreateMemory(CL_MEM_READ_WRITE,
				sizeof(T) * filterElementCount));

			biases = move(
				this->context->CreateMemory(CL_MEM_READ_WRITE,
				sizeof(T) * biasElementCount));

			random_device tempDevice;
			mt19937 mt(tempDevice());

			//TODO: The initial weight values could be something to tweak
			uniform_real_distribution<double> uniformDistribution(-0.1, 0.1);

			vector<T> initialWeightValues;
			initialWeightValues.resize(filterElementCount);
			for (int i = 0; i < filterElementCount; i++)
				initialWeightValues[i] = static_cast<T>(uniformDistribution(mt));

			vector<T> initialBiasValues;
			initialBiasValues.resize(biasElementCount);
			for (int i = 0; i < biasElementCount; i++)
				initialBiasValues[i] = static_cast<T>(uniformDistribution(mt));

			//Since this is initialization, we don't really care about which device and device queue we are using
			OCLDevice* device = this->context->GetDevices()[0];
			device->WriteMemory(filters.get(), sizeof(T) * initialWeightValues.size(),
				initialWeightValues.data(), 0, false);
			device->WriteMemory(biases.get(), sizeof(T) * initialBiasValues.size(),
				initialBiasValues.data(), 0, false);
			device->WaitForDeviceQueue(0);
		}

		template<class T>
		void ConvolutionLayer<T>::InterlockFinalized()
		{
			InitializeParameters();
			InitializePrograms();
		}

		template<class T>
		void ConvolutionLayer<T>::InitializePrograms()
		{
			vector<OCLDevice*> devices = this->context->GetDevices();
			auto backActivationFunction = this->BackPropActivationFunction();
			auto activationFunction = convolutionConfig.ActivationFunction();

			LayerDataDescription firstInputData =
				this->InForwardPropDataDescriptions()[0];

			//An intermediate cache where we store summed images
			summaryCache = this->context->CreateMemory(CL_MEM_READ_WRITE,
				sizeof(T) * firstInputData.Width * firstInputData.Height);

			for (auto device : devices)
			{
				unique_ptr<OCLProgram> program(new OCLProgram());

				program->SetUseRelaxedMath(convolutionConfig.UseRelaxedMath());
				if (convolutionConfig.ComputationPrecision() == MatunaHalfPrecision)
					program->AddDefine("HALF_MATH");
				else if (convolutionConfig.ComputationPrecision() == MatunaNativePrecision)
					program->AddDefine("NATIVE_MATH");

				program->AddIncludePath(OCLProgram::DefaultSourceLocation);

				if (is_same<cl_double, T>::value) 
					program->AddDefine("DOUBLE_PRECISION");

				if (backActivationFunction == MatunaSigmoidActivation)
					program->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_SIGMOID");
				else if (backActivationFunction == MatunaTanhActivation)
					program->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_TANH");
				else if (backActivationFunction == MatunaSoftMaxActivation)
					throw invalid_argument("Softmax is not allowed in a convolution layer at the moment");

				if (activationFunction == MatunaSigmoidActivation)
					program->AddDefine("MATUNA_ACTIVATION_SIGMOID");
				else if (activationFunction == MatunaTanhActivation)
					program->AddDefine("MATUNA_ACTIVATION_TANH");
				else if (activationFunction == MatunaSoftMaxActivation)
					program->AddDefine("MATUNA_ACTIVATION_SOFTMAX");

				program->SetName("ConvolutionLayerProgram" + Converter::ConvertToString(program->InstanceCount()));


				//Creates the kernels and attaches them to the program
				InitializeConvolutionKernel(device, program.get());
				InitializeSumAllUnitsKernel(device, program.get());
				InitializeBackPropConvolutionKernel(device, program.get());
				InitializeZeroBorderKernel(device, program.get());
				InitializeMultiplyAllUnitsKernel(device, program.get());
				InitializeSumUnitKernel(device, program.get());
				InitializeMultiplyWithOffsetKernel(device, program.get());

				//When the kernels are attached to the program, we may then attach the program to the context
				vector<OCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AttachProgram(move(program), oneDeviceVector);

				//Set all the filters and biases
				deviceAndConvolutionKernels[device]->SetMemoryArg(filters.get(), 2);
				deviceAndConvolutionKernels[device]->SetMemoryArg(biases.get(), 3);
				deviceAndBackConvolutionKernels[device]->SetMemoryArg(filters.get(), 2);
			}

		}

		template<class T>
		void ConvolutionLayer<T>::InitializeConvolutionKernel(OCLDevice* device, OCLProgram* program)
		{

			string convolutionKernelPath = Path::Combine(OCLProgram::DefaultSourceLocation, "ConvolutionKernel.cl");

			LayerDataDescription firstInputData =
				this->InForwardPropDataDescriptions()[0];
			LayerMemoryDescription firstOutputMemDesc =
				this->OutForwardPropMemoryDescriptions()[0];
			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];

			auto deviceInfo = device->DeviceInfo();
			auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(convolutionKernelPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("ConvolutionKernel");

			if (maximumConstantBufferSize > filters->ByteSize())
			{
				kernel->AddDefine(convolutionKernelPath, "CONSTANT_FILTERS");
				maximumConstantBufferSize -= filters->ByteSize();
			}

			auto byteSize = firstInputData.TotalUnits() * sizeof(T);
			if (maximumConstantBufferSize > byteSize)
			{
				kernel->AddDefine(convolutionKernelPath, "CONSTANT_INPUT");
				maximumConstantBufferSize -= byteSize;
			}

			if (maximumConstantBufferSize > biases->ByteSize())
			{
				kernel->AddDefine(convolutionKernelPath, "CONSTANT_BIAS");
				maximumConstantBufferSize -= biases->ByteSize();
			}

			kernel->AddGlobalSize(firstOutputData.Width);
			kernel->AddGlobalSize(firstOutputData.Height);
			kernel->AddGlobalSize(firstOutputData.Units);

			kernel->AddDefineSubsitute(convolutionKernelPath, "FILTER_WIDTH", convolutionConfig.FilterWidth());
			kernel->AddDefineSubsitute(convolutionKernelPath, "FILTER_HEIGHT", convolutionConfig.FilterHeight());
			kernel->AddDefineSubsitute(convolutionKernelPath, "INPUT_UNIT_MEMORY_WIDTH_OFFSET", 0);
			kernel->AddDefineSubsitute(convolutionKernelPath, "INPUT_UNIT_MEMORY_HEIGHT_OFFSET", 0);
			kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_UNIT_MEMORY_WIDTH_OFFSET", firstOutputMemDesc.WidthOffset);
			kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET", firstOutputMemDesc.HeightOffset);
			kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_UNIT_OFFSET", firstOutputMemDesc.UnitOffset);
			kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_UNIT_MEMORY_WIDTH", firstOutputMemDesc.Width);
			kernel->AddDefineSubsitute(convolutionKernelPath, "INPUT_UNIT_MEMORY_WIDTH", firstInputData.Width);
			kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_UNIT_MEMORY_ELEMENTS", firstOutputMemDesc.Width * firstOutputMemDesc.Height);
			kernel->AddDefineSubsitute(convolutionKernelPath, "FILTER_UNIT_ELEMENTS", convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight());

			deviceAndConvolutionKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeSumAllUnitsKernel(OCLDevice* device, OCLProgram* program)
		{

			string sumAllUnitsPath = Path::Combine(OCLProgram::DefaultSourceLocation, "SumAllUnitsKernel.cl");

			LayerMemoryDescription firstInputMemDesc =
				this->InForwardPropMemoryDescriptions()[0];

			LayerDataDescription firstInputData =
				this->InForwardPropDataDescriptions()[0];

			auto deviceInfo = device->DeviceInfo();
			auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(sumAllUnitsPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("SumAllUnitsKernel");

			auto totalInputMemorySize = firstInputMemDesc.TotalMemory() * sizeof(T);
			if (maximumConstantBufferSize > totalInputMemorySize)
			{
				kernel->AddDefine(sumAllUnitsPath, "CONSTANT_INPUT");
				maximumConstantBufferSize -= totalInputMemorySize;
			}

			kernel->AddGlobalSize(firstInputData.Width);
			kernel->AddGlobalSize(firstInputData.Height);

			kernel->AddDefineSubsitute(sumAllUnitsPath, "UNIT_LIMIT", firstInputData.Units + firstInputMemDesc.UnitOffset);
			kernel->AddDefineSubsitute(sumAllUnitsPath, "INPUT_UNIT_OFFSET", firstInputMemDesc.UnitOffset);
			kernel->AddDefineSubsitute(sumAllUnitsPath, "INPUT_UNIT_MEMORY_WIDTH_OFFSET", firstInputMemDesc.WidthOffset);
			kernel->AddDefineSubsitute(sumAllUnitsPath, "INPUT_UNIT_MEMORY_HEIGHT_OFFSET", firstInputMemDesc.HeightOffset);
			kernel->AddDefineSubsitute(sumAllUnitsPath, "OUTPUT_UNIT_MEMORY_WIDTH_OFFSET", 0);
			kernel->AddDefineSubsitute(sumAllUnitsPath, "OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET", 0);
			kernel->AddDefineSubsitute(sumAllUnitsPath, "INPUT_UNIT_MEMORY_WIDTH", firstInputMemDesc.Width);
			kernel->AddDefineSubsitute(sumAllUnitsPath, "OUTPUT_UNIT_MEMORY_WIDTH", firstInputData.Width);
			kernel->AddDefineSubsitute(sumAllUnitsPath, "INPUT_UNIT_MEMORY_ELEMENTS", firstInputMemDesc.Width * firstInputMemDesc.Height);

			deviceAndSumKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeBackPropConvolutionKernel(OCLDevice* device, OCLProgram* program)
		{		

			string backConvolutionPath = Path::Combine(OCLProgram::DefaultSourceLocation, "BackPropConvolutionKernel.cl");

			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];

			LayerMemoryDescription firstInMemDesc =
				this->InBackPropMemoryDescriptions()[0];

			LayerDataDescription firstInputData =
				this->InForwardPropDataDescriptions()[0];

			auto deviceInfo = device->DeviceInfo();
			auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(backConvolutionPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("BackPropConvolutionKernel");

			if (maximumConstantBufferSize > filters->ByteSize())
			{
				kernel->AddDefine(backConvolutionPath, "CONSTANT_FILTERS");
				maximumConstantBufferSize -= filters->ByteSize();
			}

			auto byteSize = firstInMemDesc.TotalMemory() * sizeof(T);
			if (maximumConstantBufferSize > byteSize)
			{
				kernel->AddDefine(backConvolutionPath, "CONSTANT_INPUT_DELTA");
				maximumConstantBufferSize -= byteSize;
			}

			kernel->AddGlobalSize(firstInputData.Width);
			kernel->AddGlobalSize(firstInputData.Height);

			kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_COUNT", firstOutputData.Units);
			kernel->AddDefineSubsitute(backConvolutionPath, "FILTER_WIDTH", convolutionConfig.FilterWidth());
			kernel->AddDefineSubsitute(backConvolutionPath, "FILTER_HEIGHT", convolutionConfig.FilterHeight());
			kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_OFFSET", firstInMemDesc.UnitOffset);
			kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_LIMIT", firstInMemDesc.UnitOffset + firstOutputData.Units);
			kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_MEMORY_WIDTH", firstInMemDesc.Width);
			kernel->AddDefineSubsitute(backConvolutionPath, "OUTPUT_UNIT_MEMORY_WIDTH", firstInputData.Width);
			kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_MEMORY_WIDTH_OFFSET", firstInMemDesc.WidthOffset - convolutionConfig.FilterWidth() + 1);
			kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_MEMORY_HEIGHT_OFFSET", firstInMemDesc.HeightOffset - convolutionConfig.FilterHeight() + 1);
			kernel->AddDefineSubsitute(backConvolutionPath, "OUTPUT_UNIT_MEMORY_WIDTH_OFFSET", 0);
			kernel->AddDefineSubsitute(backConvolutionPath, "OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET", 0);
			kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_MEMORY_ELEMENTS", firstInMemDesc.Width * firstInMemDesc.Height);
			kernel->AddDefineSubsitute(backConvolutionPath, "FILTER_UNIT_ELEMENTS", convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight());

			deviceAndBackConvolutionKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeZeroBorderKernel(OCLDevice* device, OCLProgram* program)
		{
			string zeroKernelPath = Path::Combine(OCLProgram::DefaultSourceLocation, "ZeroBorderKernel.cl");

			LayerMemoryDescription firstInBackMemDesc =
				this->InBackPropMemoryDescriptions()[0];

			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(zeroKernelPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("ZeroBorderKernel");

			int borderHorizontalSize = convolutionConfig.FilterWidth() - 1;
			int borderVerticalSize = convolutionConfig.FilterHeight() - 1;

			int borderStartLeft = firstInBackMemDesc.WidthOffset
				- borderHorizontalSize;
			if (borderStartLeft < 0)
				throw runtime_error("The memory / data descriptions are invalid");

			int borderStartRight = firstInBackMemDesc.WidthOffset
				+ firstOutputData.Width;

			int borderStartUp = firstInBackMemDesc.HeightOffset
				- borderVerticalSize;
			if (borderStartUp < 0)
				throw runtime_error("The memory / data descriptions are invalid");

			int borderStartDown = firstInBackMemDesc.HeightOffset
				+ firstOutputData.Height;

			kernel->AddGlobalSize(firstOutputData.Units);

			kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_UNIT_MEMORY_ELEMENTS", firstInBackMemDesc.Width * firstInBackMemDesc.Height);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_START_LEFT", borderStartLeft);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_START_RIGHT", borderStartRight);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_START_UP", borderStartUp);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_START_DOWN", borderStartDown);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_LIMIT_LEFT", borderStartLeft + borderHorizontalSize - 1);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_LIMIT_RIGHT", borderStartRight + borderHorizontalSize - 1);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_LIMIT_UP", borderStartUp + borderVerticalSize - 1);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_LIMIT_DOWN", borderStartDown + borderVerticalSize - 1);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_SIZE_HORIZONTAL", borderHorizontalSize);
			kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_SIZE_VERTICAL", borderVerticalSize);
			kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_UNIT_OFFSET", firstInBackMemDesc.UnitOffset);
			kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_UNIT_WIDTH", firstOutputData.Width);
			kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_UNIT_HEIGHT", firstOutputData.Height);
			kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_UNIT_MEMORY_WIDTH", firstInBackMemDesc.Width);

			deviceAndZeroKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeMultiplyAllUnitsKernel(OCLDevice* device, OCLProgram* program)
		{


			string multiplyKernelPath = Path::Combine(OCLProgram::DefaultSourceLocation, "MultiplyAllUnitsKernel.cl");

			LayerMemoryDescription firstOutputBackMemDesc =
				this->OutBackPropMemoryDescriptions()[0];
			LayerMemoryDescription firstInForwardMemDesc =
				this->InForwardPropMemoryDescriptions()[0];
			LayerDataDescription firstInputData =
				this->InForwardPropDataDescriptions()[0];

			auto deviceInfo = device->DeviceInfo();
			auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(multiplyKernelPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("MultiplyAllUnitsKernel");

			auto deltaBytes = firstOutputBackMemDesc.TotalMemory() * sizeof(T);
			auto inputBytes = firstInForwardMemDesc.TotalMemory() * sizeof(T);

			if (maximumConstantBufferSize > deltaBytes)
			{
				kernel->AddDefine(multiplyKernelPath, "CONSTANT_INPUT_DELTA");
				maximumConstantBufferSize -= deltaBytes;
			}

			if (maximumConstantBufferSize > inputBytes)
			{
				kernel->AddDefine(multiplyKernelPath, "CONSTANT_INPUT");
				maximumConstantBufferSize -= inputBytes;
			}

			kernel->AddGlobalSize(firstInputData.Width);
			kernel->AddGlobalSize(firstInputData.Height);
			kernel->AddGlobalSize(firstInputData.Units);

			kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_UNIT_MEMORY_ELEMENTS", firstInForwardMemDesc.Width * firstInForwardMemDesc.Height);
			kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_UNIT_MEMORY_ELEMENTS", firstOutputBackMemDesc.Width * firstOutputBackMemDesc.Height);
			kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_DELTA_UNIT_MEMORY_WIDTH", firstInputData.Width);
			kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_UNIT_MEMORY_WIDTH", firstOutputBackMemDesc.Width);
			kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_UNIT_MEMORY_WIDTH", firstInForwardMemDesc.Width);
			kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_DELTA_UNIT_MEMORY_WIDTH_OFFSET", 0);
			kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_DELTA_UNIT_MEMORY_HEIGHT_OFFSET", 0);
			kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_UNIT_MEMORY_WIDTH_OFFSET", firstOutputBackMemDesc.WidthOffset);
			kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET", firstOutputBackMemDesc.HeightOffset);
			kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_UNIT_OFFSET", firstOutputBackMemDesc.UnitOffset);
			kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_UNIT_MEMORY_WIDTH_OFFSET", firstInForwardMemDesc.WidthOffset);
			kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_UNIT_MEMORY_HEIGHT_OFFSET", firstInForwardMemDesc.HeightOffset);
			kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_UNIT_OFFSET", firstInForwardMemDesc.UnitOffset);

			deviceAndMultiplyKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeSumUnitKernel(OCLDevice* device, OCLProgram* program)
		{

			string sumUnitPath = Path::Combine(OCLProgram::DefaultSourceLocation, "SumUnitKernel.cl");

			LayerMemoryDescription firstInBackMemDesc =
				this->InBackPropMemoryDescriptions()[0];

			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];

			auto deviceInfo = device->DeviceInfo();
			auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(sumUnitPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("SumUnitKernel");

			auto deltaBytes = firstInBackMemDesc.TotalMemory() * sizeof(T);

			if (maximumConstantBufferSize > deltaBytes)
			{
				kernel->AddDefine(sumUnitPath, "CONSTANT_INPUT");
				maximumConstantBufferSize -= deltaBytes;
			}

			kernel->AddGlobalSize(convolutionConfig.FilterCount());

			kernel->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_MEMORY_ELEMENTS", firstInBackMemDesc.Width * firstInBackMemDesc.Height);
			kernel->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_MEMORY_WIDTH", firstInBackMemDesc.Width);
			kernel->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_OFFSET", firstInBackMemDesc.UnitOffset);
			kernel->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_MEMORY_WIDTH_OFFSET", firstInBackMemDesc.WidthOffset);
			kernel->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_MEMORY_HEIGHT_OFFSET", firstInBackMemDesc.HeightOffset);
			kernel->AddDefineSubsitute(sumUnitPath, "WIDTH_LIMIT", firstInBackMemDesc.WidthOffset +firstOutputData.Width);
			kernel->AddDefineSubsitute(sumUnitPath, "HEIGHT_LIMIT", firstInBackMemDesc.HeightOffset + firstOutputData.Height);
			kernel->AddDefineSubsitute(sumUnitPath, "OUTPUT_OFFSET", 0); //Since we are splitting the gradient for this kernel

			deviceAndSumUnitKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeMultiplyWithOffsetKernel(OCLDevice* device, OCLProgram* program)
		{

			string multiplyOffsetPath = Path::Combine(OCLProgram::DefaultSourceLocation, "MultiplyWithOffsetKernel.cl");

			LayerMemoryDescription firstInBackMemDesc =
				this->InBackPropMemoryDescriptions()[0];

			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];

			LayerDataDescription firstInputData =
				this->InForwardPropDataDescriptions()[0];

			LayerMemoryDescription firstInForwardMemDesc =
				this->InForwardPropMemoryDescriptions()[0];

			auto deviceInfo = device->DeviceInfo();
			auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(multiplyOffsetPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("MultiplyWithOffsetKernel");

			auto deltaBytes = firstInBackMemDesc.TotalMemory() * sizeof(T);

			if (maximumConstantBufferSize > deltaBytes)
			{
				kernel->AddDefine(multiplyOffsetPath, "CONSTANT_INPUT_DELTA");
				maximumConstantBufferSize -= deltaBytes;
			}

			auto inputBytes = firstInForwardMemDesc.TotalMemory() * sizeof(T);
			if (maximumConstantBufferSize > inputBytes)
			{
				kernel->AddDefine(multiplyOffsetPath, "CONSTANT_INPUT");
				maximumConstantBufferSize -= inputBytes;
			}

			kernel->AddGlobalSize(convolutionConfig.FilterWidth());
			kernel->AddGlobalSize(convolutionConfig.FilterHeight());
			kernel->AddGlobalSize(convolutionConfig.FilterCount());

			kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_UNIT_MEMORY_WIDTH", firstInBackMemDesc.Width);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_UNIT_MEMORY_WIDTH", convolutionConfig.FilterWidth());
			kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_UNIT_MEMORY_WIDTH", firstInputData.Width);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_UNIT_MEMORY_WIDTH_OFFSET", 0);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_UNIT_MEMORY_HEIGHT_OFFSET", 0);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_UNIT_MEMORY_WIDTH_OFFSET", firstInBackMemDesc.WidthOffset);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_UNIT_MEMORY_HEIGHT_OFFSET", firstInBackMemDesc.HeightOffset);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_UNIT_OFFSET", firstInBackMemDesc.UnitOffset);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "WIDTH_LIMIT", firstOutputData.Width);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "HEIGHT_LIMIT", firstOutputData.Height);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_UNIT_MEMORY_ELEMENTS", convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight());
			kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_UNIT_MEMORY_ELEMENTS", firstInBackMemDesc.Height * firstInBackMemDesc.Width);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_UNIT_MEMORY_WIDTH_OFFSET", 0);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET", 0);
			kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_UNIT_OFFSET", 0);

			deviceAndMultiplyWithOffsetKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueForwardPropagation(OCLDevice* device,
			int queueIndex, OCLMemory* previousInput, OCLMemory* output,
			bool blocking)
		{
			auto sumAllUnitsKernel = deviceAndSumKernels[device];
			auto convolutionKernel = deviceAndConvolutionKernels[device];
			sumAllUnitsKernel->SetMemoryArg(previousInput, 0);
			sumAllUnitsKernel->SetMemoryArg(summaryCache.get(), 1);
			device->ExecuteKernel(sumAllUnitsKernel, queueIndex, false);
			convolutionKernel->SetMemoryArg(summaryCache.get(), 0);
			convolutionKernel->SetMemoryArg(output, 1);
			device->ExecuteKernel(convolutionKernel, queueIndex, blocking);
		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueBackPropagation(OCLDevice* device,
			int queueIndex, OCLMemory* previousInput, OCLMemory* delta,
			OCLMemory* deltaOutput, bool blocking)
		{
			auto zeroKernel = deviceAndZeroKernels[device];
			zeroKernel->SetMemoryArg(delta, 0);
			auto convolutionKernel = deviceAndBackConvolutionKernels[device];
			convolutionKernel->SetMemoryArg(delta, 0);
			convolutionKernel->SetMemoryArg(summaryCache.get(), 1);
			auto multiplyKernel = deviceAndMultiplyKernels[device];
			multiplyKernel->SetMemoryArg(previousInput, 0);
			multiplyKernel->SetMemoryArg(summaryCache.get(), 1);
			multiplyKernel->SetMemoryArg(deltaOutput, 2);
			device->ExecuteKernel(zeroKernel, queueIndex, false);
			device->ExecuteKernel(convolutionKernel, queueIndex, false);
			device->ExecuteKernel(multiplyKernel, queueIndex, blocking);
		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* delta, vector<OCLMemory*> gradient, bool blocking)
		{

			if (gradient.size() != 2)
				throw invalid_argument("The gradient size is not valid");

			if (gradient[0]->ByteSize() / sizeof(T) != static_cast<size_t>(convolutionConfig.FilterCount() * convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight()))
				throw invalid_argument("The first gradient does not contain the correct amount of memory");

			if (gradient[1]->ByteSize() / sizeof(T) != static_cast<size_t>(convolutionConfig.FilterCount()))
				throw invalid_argument("The second gradient does not contain the correct amount of memory");

			auto sumAllUnitsKernel = deviceAndSumKernels[device];
			sumAllUnitsKernel->SetMemoryArg(previousInput, 0);
			sumAllUnitsKernel->SetMemoryArg(summaryCache.get(), 1);
			auto multiplyWithOffsetKernel = deviceAndMultiplyWithOffsetKernels[device];
			multiplyWithOffsetKernel->SetMemoryArg(summaryCache.get(), 0);
			multiplyWithOffsetKernel->SetMemoryArg(delta, 1);
			multiplyWithOffsetKernel->SetMemoryArg(gradient[0], 2);
			auto sumUnitKernel = deviceAndSumUnitKernels[device];
			sumUnitKernel->SetMemoryArg(delta, 0);
			sumUnitKernel->SetMemoryArg(gradient[1], 1);

			device->ExecuteKernel(sumAllUnitsKernel, queueIndex, false);
			device->ExecuteKernel(multiplyWithOffsetKernel, queueIndex, false);
			device->ExecuteKernel(sumUnitKernel, queueIndex, blocking);
		}

		template<class T>
		vector<size_t> ConvolutionLayer<T>::GetMultipleParameterCount()
		{
			vector<size_t> result;
			result.push_back(convolutionConfig.FilterCount() * convolutionConfig.FilterWidth()
				* convolutionConfig.FilterHeight());
			result.push_back(convolutionConfig.FilterCount());
			return result;
		}

		template<class T>
		vector<OCLMemory*> ConvolutionLayer<T>::GetParameters()
		{
			vector<OCLMemory*> result;

			result.push_back(filters.get());
			result.push_back(biases.get());

			return result;
		}

		template<class T>
		void ConvolutionLayer<T>::GetParameters(T* parameters, OCLDevice* device,
			int queueIndex, bool blocking)
		{
			device->ReadMemory(filters.get(), filters->ByteSize(), parameters,
				queueIndex, blocking);

			auto biasPosition = parameters
				+ convolutionConfig.FilterCount() * convolutionConfig.FilterWidth()
				* convolutionConfig.FilterHeight();

			device->ReadMemory(biases.get(), biases->ByteSize(), biasPosition,
				queueIndex, blocking);
		}

		template<class T>
		void ConvolutionLayer<T>::SetParameters(T* parameters, OCLDevice* device,
			int queueIndex, bool blocking)
		{
			device->WriteMemory(filters.get(), filters->ByteSize(), parameters,
				queueIndex, blocking);

			auto biasPosition = parameters
				+ convolutionConfig.FilterCount() * convolutionConfig.FilterWidth()
				* convolutionConfig.FilterHeight();

			device->WriteMemory(biases.get(), biases->ByteSize(), biasPosition,
				queueIndex, blocking);
		}

		template<class T>
		size_t ConvolutionLayer<T>::GetParameterCount()
		{
			return convolutionConfig.FilterCount() * convolutionConfig.FilterWidth()
				* convolutionConfig.FilterHeight() + convolutionConfig.FilterCount();
		}

		template class ConvolutionLayer<cl_float> ;
		template class ConvolutionLayer<cl_double> ;

	} /* namespace MachineLearning */
} /* namespace Matuna */
