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
		ConvolutionLayer<T>::~ConvolutionLayer()
		{

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
			uniform_real_distribution<T> uniformDistribution(-0.1, 0.1);

			vector<T> initialWeightValues;
			initialWeightValues.resize(filterElementCount);
			for (int i = 0; i < filterElementCount; i++)
				initialWeightValues[i] = uniformDistribution(mt);

			vector<T> initialBiasValues;
			initialBiasValues.resize(biasElementCount);
			for (int i = 0; i < biasElementCount; i++)
				initialBiasValues[i] = uniformDistribution(mt);

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

			//Make sure the type we want to execute is supported on the device.
			vector<OCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices) {
				auto deviceInfo = device->DeviceInfo();
				if (is_same<cl_double, T>::value) {
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				} else if (is_same<cl_float, T>::value) {
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				} else
					throw runtime_error(
					"The template argument does not match the supported arguments");
			}

			InitializeParameters();
			//HACK---------------------
			unordered_map<OCLDevice*, unique_ptr<OCLProgram>> fixThisPrograms;
			//END HACK-----------------

			unordered_map<OCLDevice*, unique_ptr<OCLProgram>> programs;
			auto backActivationFunction = this->BackPropActivationFunction();
			auto activationFunction = convolutionConfig.ActivationFunction();
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

				program->SetName("ConvolutionLayerProgram" + to_string(program->InstanceCount()));
				programs.insert(make_pair(device, move(program)));

				//HACK---------------------

				unique_ptr<OCLProgram> fixThisProgram(new OCLProgram());

				fixThisProgram->SetUseRelaxedMath(convolutionConfig.UseRelaxedMath());
				if (convolutionConfig.ComputationPrecision() == MatunaHalfPrecision)
					fixThisProgram->AddDefine("HALF_MATH");
				else if (convolutionConfig.ComputationPrecision() == MatunaNativePrecision)
					fixThisProgram->AddDefine("NATIVE_MATH");

				fixThisProgram->AddIncludePath(OCLProgram::DefaultSourceLocation);

				if (is_same<cl_double, T>::value) 
					fixThisProgram->AddDefine("DOUBLE_PRECISION");

				if (backActivationFunction == MatunaSigmoidActivation)
					fixThisProgram->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_SIGMOID");
				else if (backActivationFunction == MatunaTanhActivation)
					fixThisProgram->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_TANH");
				else if (backActivationFunction == MatunaSoftMaxActivation)
					throw invalid_argument("Softmax is not allowed in a convolution layer at the moment");

				if (activationFunction == MatunaSigmoidActivation)
					fixThisProgram->AddDefine("MATUNA_ACTIVATION_SIGMOID");
				else if (activationFunction == MatunaTanhActivation)
					fixThisProgram->AddDefine("MATUNA_ACTIVATION_TANH");
				else if (activationFunction == MatunaSoftMaxActivation)
					fixThisProgram->AddDefine("MATUNA_ACTIVATION_SOFTMAX");

				fixThisProgram->SetName("fixThisProgram" + to_string(program->InstanceCount()));
				fixThisPrograms.insert(make_pair(device, move(fixThisProgram)));

				//END HACK-----------------

			}

			InitializeProgram(programs);
			//Attaching and compiling all the programs 
			for (auto device : devices)
			{

				vector<OCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);

				//HACK---------------------
				LayerMemoryDescription firstInBackMemDesc =
					this->InBackPropMemoryDescriptions()[0];
				LayerDataDescription firstOutputData =
					this->outForwardPropDataDescriptions[0];
				string sumUnitPath = Path::Combine(OCLProgram::DefaultSourceLocation, "SumUnitKernel.cl");
				auto deviceInfo = device->DeviceInfo();
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

				unique_ptr<LayerKernel<T>> kernel2(new LayerKernel<T>());
				kernel2->AddSourcePath(sumUnitPath);
				kernel2->AddIncludePath(OCLProgram::DefaultSourceLocation);
				kernel2->SetKernelName("SumUnitKernel");

				auto deltaBytes = firstInBackMemDesc.TotalMemory() * sizeof(T);

				if (maximumConstantBufferSize > deltaBytes)
				{
					kernel2->AddDefine(sumUnitPath, "CONSTANT_INPUT");
					maximumConstantBufferSize -= deltaBytes;
				}

				kernel2->AddGlobalSize(convolutionConfig.FilterCount());

				kernel2->AddDefineSubsitute(sumUnitPath, "INPUT_STRIDE", to_string(firstInBackMemDesc.Width));
				kernel2->AddDefineSubsitute(sumUnitPath, "INPUT_WIDTH_OFFSET", to_string(firstInBackMemDesc.WidthOffset));
				kernel2->AddDefineSubsitute(sumUnitPath, "WIDTH_LIMIT", to_string(firstInBackMemDesc.WidthOffset +firstOutputData.Width));
				kernel2->AddDefineSubsitute(sumUnitPath, "HEIGHT_LIMIT", to_string(firstInBackMemDesc.HeightOffset + firstOutputData.Height));
				kernel2->AddDefineSubsitute(sumUnitPath, "INPUT_HEIGHT_OFFSET", to_string(firstInBackMemDesc.HeightOffset));
				kernel2->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_OFFSET", to_string(firstInBackMemDesc.UnitOffset));
				kernel2->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstInBackMemDesc.Width * firstInBackMemDesc.Height));
				kernel2->AddDefineSubsitute(sumUnitPath, "OUTPUT_OFFSET", to_string(convolutionConfig.FilterHeight() * convolutionConfig.FilterWidth() * convolutionConfig.FilterCount()));

				deviceAndSumUnitKernels2.insert(make_pair(device, kernel2.get()));
				fixThisPrograms[device]->AttachKernel(move(kernel2));

				//END HACK-----------------

				context->AttachProgram(move(programs[device]), oneDeviceVector);
				context->AttachProgram(move(fixThisPrograms[device]), oneDeviceVector);

				//Set all the filters and biases
				deviceAndConvolutionKernels[device]->SetMemoryArg(filters.get(), 2);
				deviceAndConvolutionKernels[device]->SetMemoryArg(biases.get(), 3);
				deviceAndBackConvolutionKernels[device]->SetMemoryArg(filters.get(), 2);
			}

		}

		template<class T>
		void ConvolutionLayer<T>::InitializeProgram(unordered_map<OCLDevice*, unique_ptr<OCLProgram>>& programs)
		{
			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];
			LayerMemoryDescription firstOutputBackMemDesc =
				this->OutBackPropMemoryDescriptions()[0];
			LayerMemoryDescription firstInBackMemDesc =
				this->InBackPropMemoryDescriptions()[0];
			LayerMemoryDescription firstInForwardMemDesc =
				this->InForwardPropMemoryDescriptions()[0];
			LayerDataDescription firstInputData =
				this->InForwardPropDataDescriptions()[0];

			LayerMemoryDescription firstOutputMemDesc =
				this->OutForwardPropMemoryDescriptions()[0];

			LayerMemoryDescription firstInputMemDesc =
				this->InForwardPropMemoryDescriptions()[0];

			LayerMemoryDescription firstInMemDesc =
				this->InBackPropMemoryDescriptions()[0];

			string convolutionKernelPath = Path::Combine(OCLProgram::DefaultSourceLocation, "ConvolutionKernel.cl");
			string sumAllUnitsPath = Path::Combine(OCLProgram::DefaultSourceLocation, "SumAllUnitsKernel.cl");
			string backConvolutionPath = Path::Combine(OCLProgram::DefaultSourceLocation, "BackPropConvolutionKernel.cl");
			string zeroKernelPath = Path::Combine(OCLProgram::DefaultSourceLocation, "ZeroBorderKernel.cl");
			string multiplyKernelPath = Path::Combine(OCLProgram::DefaultSourceLocation, "MultiplyAllUnitsKernel.cl");
			string sumUnitPath = Path::Combine(OCLProgram::DefaultSourceLocation, "SumUnitKernel.cl");
			string multiplyOffsetPath = Path::Combine(OCLProgram::DefaultSourceLocation, "MultiplyWithOffsetKernel.cl");

			vector<OCLDevice*> devices = this->context->GetDevices();

			for (auto device : devices)
			{
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

				kernel->AddDefineSubsitute(convolutionKernelPath, "FILTER_WIDTH", to_string(convolutionConfig.FilterWidth()));
				kernel->AddDefineSubsitute(convolutionKernelPath, "FILTER_HEIGHT", to_string(convolutionConfig.FilterHeight()));
				kernel->AddDefineSubsitute(convolutionKernelPath, "INPUT_OFFSET_WIDTH", to_string(0));
				kernel->AddDefineSubsitute(convolutionKernelPath, "INPUT_OFFSET_HEIGHT", to_string(0));
				kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_OFFSET_WIDTH", to_string(firstOutputMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_OFFSET_HEIGHT", to_string(firstOutputMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_OFFSET_UNIT", to_string(firstOutputMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_WIDTH", to_string(firstOutputMemDesc.Width));
				kernel->AddDefineSubsitute(convolutionKernelPath, "INPUT_WIDTH", to_string(firstInputData.Width));
				kernel->AddDefineSubsitute(convolutionKernelPath, "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstOutputMemDesc.Width * firstOutputMemDesc.Height));
				kernel->AddDefineSubsitute(convolutionKernelPath, "FILTER_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight()));

				deviceAndConvolutionKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			summaryCache = this->context->CreateMemory(CL_MEM_READ_WRITE,
				sizeof(T) * firstInputData.Width * firstInputData.Height);
			for (auto device : devices)
			{
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

				kernel->AddDefineSubsitute(sumAllUnitsPath, "UNIT_COUNT_INC_PADDING", to_string(firstInputData.Units + firstInputMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(sumAllUnitsPath, "UNIT_INPUT_OFFSET", to_string(firstInputMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(sumAllUnitsPath, "WIDTH_INPUT_OFFSET", to_string(firstInputMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(sumAllUnitsPath, "HEIGHT_INPUT_OFFSET", to_string(firstInputMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(sumAllUnitsPath, "WIDTH_OUTPUT_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(sumAllUnitsPath, "HEIGHT_OUTPUT_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(sumAllUnitsPath, "WIDTH_INPUT", to_string(firstInputMemDesc.Width));
				kernel->AddDefineSubsitute(sumAllUnitsPath, "WIDTH_OUTPUT", to_string(firstInputData.Width));
				kernel->AddDefineSubsitute(sumAllUnitsPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstInputMemDesc.Width * firstInputMemDesc.Height));

				deviceAndSumKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			for (auto device : devices)
			{
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

				kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_COUNT", to_string(firstOutputData.Units));
				kernel->AddDefineSubsitute(backConvolutionPath, "FILTER_WIDTH", to_string(convolutionConfig.FilterWidth()));
				kernel->AddDefineSubsitute(backConvolutionPath, "FILTER_HEIGHT", to_string(convolutionConfig.FilterHeight()));
				kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_OFFSET", to_string(firstInMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_LIMIT", to_string(firstInMemDesc.UnitOffset + firstOutputData.Units));
				kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_STRIDE", to_string(firstInMemDesc.Width));
				kernel->AddDefineSubsitute(backConvolutionPath, "OUTPUT_STRIDE", to_string(firstInputData.Width));
				kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_WIDTH_OFFSET", to_string(firstInMemDesc.WidthOffset - convolutionConfig.FilterWidth() + 1));
				kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_HEIGHT_OFFSET", to_string(firstInMemDesc.HeightOffset - convolutionConfig.FilterHeight() + 1));
				kernel->AddDefineSubsitute(backConvolutionPath, "OUTPUT_WIDTH_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(backConvolutionPath, "OUTPUT_HEIGHT_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(backConvolutionPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstInMemDesc.Width * firstInMemDesc.Height));
				kernel->AddDefineSubsitute(backConvolutionPath, "FILTER_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight()));

				deviceAndBackConvolutionKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			for (auto device : devices)
			{
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

				kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstInBackMemDesc.Width * firstInBackMemDesc.Height));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_START_LEFT", to_string(borderStartLeft));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_START_RIGHT", to_string(borderStartRight));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_START_UP", to_string(borderStartUp));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_START_DOWN", to_string(borderStartDown));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_LIMIT_LEFT", to_string(borderStartLeft + borderHorizontalSize - 1));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_LIMIT_RIGHT", to_string(borderStartRight + borderHorizontalSize - 1));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_LIMIT_UP", to_string(borderStartUp + borderVerticalSize - 1));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_LIMIT_DOWN", to_string(borderStartDown + borderVerticalSize - 1));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_SIZE_HORIZONTAL", to_string(borderHorizontalSize));
				kernel->AddDefineSubsitute(zeroKernelPath, "BORDER_SIZE_VERTICAL", to_string(borderVerticalSize));
				kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_UNIT_OFFSET", to_string(firstInBackMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_DATA_WIDTH", to_string(firstOutputData.Width));
				kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_DATA_HEIGHT", to_string(firstOutputData.Height));
				kernel->AddDefineSubsitute(zeroKernelPath, "INPUT_STRIDE", to_string(firstInBackMemDesc.Width));

				deviceAndZeroKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			for (auto device : devices)
			{
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

				kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstInForwardMemDesc.Width * firstInForwardMemDesc.Height));
				kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstOutputBackMemDesc.Width * firstOutputBackMemDesc.Height));
				kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_DELTA_STRIDE", to_string(firstInputData.Width));
				kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_STRIDE", to_string(firstOutputBackMemDesc.Width));
				kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_STRIDE", to_string(firstInForwardMemDesc.Width));
				kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_DELTA_WIDTH_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_DELTA_HEIGHT_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_WIDTH_OFFSET", to_string(firstOutputBackMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_HEIGHT_OFFSET", to_string(firstOutputBackMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(multiplyKernelPath, "OUTPUT_UNIT_OFFSET", to_string(firstOutputBackMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_WIDTH_OFFSET", to_string(firstInForwardMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_HEIGHT_OFFSET", to_string(firstInForwardMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(multiplyKernelPath, "INPUT_UNIT_OFFSET", to_string(firstInForwardMemDesc.UnitOffset));

				deviceAndMultiplyKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			for (auto device : devices)
			{
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

				kernel->AddDefineSubsitute(sumUnitPath, "INPUT_STRIDE", to_string(firstInBackMemDesc.Width));
				kernel->AddDefineSubsitute(sumUnitPath, "INPUT_WIDTH_OFFSET", to_string(firstInBackMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(sumUnitPath, "WIDTH_LIMIT", to_string(firstInBackMemDesc.WidthOffset +firstOutputData.Width));
				kernel->AddDefineSubsitute(sumUnitPath, "HEIGHT_LIMIT", to_string(firstInBackMemDesc.HeightOffset + firstOutputData.Height));
				kernel->AddDefineSubsitute(sumUnitPath, "INPUT_HEIGHT_OFFSET", to_string(firstInBackMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_OFFSET", to_string(firstInBackMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(sumUnitPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstInBackMemDesc.Width * firstInBackMemDesc.Height));
				kernel->AddDefineSubsitute(sumUnitPath, "OUTPUT_OFFSET", to_string(convolutionConfig.FilterHeight() * convolutionConfig.FilterWidth() * convolutionConfig.FilterCount()));

				deviceAndSumUnitKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			for (auto device : devices)
			{
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

				kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_STRIDE", to_string(firstInBackMemDesc.Width));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_STRIDE", to_string(convolutionConfig.FilterWidth()));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_STRIDE", to_string(firstInputData.Width));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_WIDTH_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_HEIGHT_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_WIDTH_OFFSET", to_string(firstInBackMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_HEIGHT_OFFSET", to_string(firstInBackMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_UNIT_OFFSET", to_string(firstInBackMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "WIDTH_LIMIT", to_string(firstOutputData.Width));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "HEIGHT_LIMIT", to_string(firstOutputData.Height));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight()));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstInBackMemDesc.Height * firstInBackMemDesc.Width));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_WIDTH_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_HEIGHT_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(multiplyOffsetPath, "OUTPUT_UNIT_OFFSET", to_string(0));

				deviceAndMultiplyWithOffsetKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}
		}

		/*
		template<class T>
		void ConvolutionLayer<T>::InitializeConvolutionKernel()
		{

		//TODO: Add a kernel for every type of output configuration
		LayerDataDescription firstOutputData =
		this->outForwardPropDataDescriptions[0];
		LayerMemoryDescription firstOutputMemDesc =
		this->OutForwardPropMemoryDescriptions()[0];
		LayerDataDescription firstInputData =
		this->InForwardPropDataDescriptions()[0];

		vector<OCLDevice*> devices = this->context->GetDevices();
		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

		//Since the sum all units kernel is not using any padding, it has to be zero here for in input description.
		//TODO: We are not using local memory for GPU devices at the moment.
		unique_ptr<ConvolutionKernel<T>> kernel(
		new ConvolutionKernel<T>(firstOutputData.Units,
		firstOutputData.Width, firstOutputData.Height,
		convolutionConfig.FilterWidth(),
		convolutionConfig.FilterHeight(), 0, 0,
		firstOutputMemDesc.WidthOffset,
		firstOutputMemDesc.HeightOffset,
		firstOutputMemDesc.UnitOffset, firstOutputMemDesc.Width,
		firstInputData.Width,
		firstOutputMemDesc.Width * firstOutputMemDesc.Height,
		convolutionConfig.FilterWidth()
		* convolutionConfig.FilterHeight(), false));

		//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
		auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

		auto byteSize = convolutionConfig.FilterWidth()
		* convolutionConfig.FilterHeight()
		* convolutionConfig.FilterCount() * sizeof(T);
		if (maximumConstantBufferSize > byteSize)
		{
		kernel->SetConstantFilters(true);
		maximumConstantBufferSize -= byteSize;
		}

		byteSize = firstInputData.TotalUnits() * sizeof(T);
		if (maximumConstantBufferSize > byteSize)
		{
		kernel->SetConstantInput(true);
		maximumConstantBufferSize -= byteSize;
		}

		byteSize = convolutionConfig.FilterCount() * sizeof(T);
		if (maximumConstantBufferSize > byteSize)
		{
		kernel->SetConstantBias(true);
		maximumConstantBufferSize -= byteSize;
		}

		kernel->SetRelaxedMath(convolutionConfig.UseRelaxedMath());
		kernel->SetActivationFunction(convolutionConfig.ActivationFunction());
		kernel->SetComputationPrecision(
		convolutionConfig.ComputationPrecision());

		kernel->InitializeCompilerOptions();

		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
		this->context->AddKernel(kernel.get());

		kernel->SetFilters(filters.get());
		kernel->SetBiases(biases.get());

		deviceAndConvolutionKernels.insert(make_pair(device, move(kernel)));
		}
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeSumAllKernel()
		{
		LayerDataDescription firstInputData =
		this->InForwardPropDataDescriptions()[0];
		LayerMemoryDescription firstInputMemDesc =
		this->InForwardPropMemoryDescriptions()[0];

		vector<OCLDevice*> devices = this->context->GetDevices();
		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

		//We are not using any padding at all in this kernel. Meaning that the convolution kernel cannot use it either
		unique_ptr<SumAllUnitsKernel<T>> kernel(
		new SumAllUnitsKernel<T>(firstInputData.Width,
		firstInputData.Height, firstInputData.Units,
		firstInputMemDesc.WidthOffset,
		firstInputMemDesc.HeightOffset,
		firstInputMemDesc.UnitOffset, firstInputMemDesc.Width,
		firstInputMemDesc.Height, 0, 0, firstInputData.Width,
		firstInputData.Height));

		summaryCache = this->context->CreateMemory(CL_MEM_READ_WRITE,
		sizeof(T) * firstInputData.Width * firstInputData.Height);

		auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
		auto totalInputMemorySize = firstInputMemDesc.TotalMemory() * sizeof(T);

		if (maximumConstantBufferSize > totalInputMemorySize)
		{
		kernel->SetUseConstantInput(true);
		maximumConstantBufferSize -= totalInputMemorySize;
		}

		kernel->SetUseRelaxedMath(convolutionConfig.UseRelaxedMath());
		kernel->InitializeCompilerOptions();
		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
		this->context->AddKernel(kernel.get());
		deviceAndSumKernels.insert(make_pair(device, move(kernel)));
		}
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeBackConvolutionKernel()
		{
		LayerDataDescription firstOutputData =
		this->outForwardPropDataDescriptions[0];
		LayerMemoryDescription firstOutputMemDesc =
		this->OutBackPropMemoryDescriptions()[0];
		LayerMemoryDescription firstInMemDesc =
		this->InBackPropMemoryDescriptions()[0];
		LayerDataDescription firstInputData =
		this->InForwardPropDataDescriptions()[0];

		vector<OCLDevice*> devices = this->context->GetDevices();
		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

		//Since the sum all units kernel is not using any padding, it has to be zero here for in input description.
		//TODO: We are not using local memory for GPU devices at the moment.
		unique_ptr<BackConvolutionKernel<T>> kernel(
		new BackConvolutionKernel<T>(firstInputData.Width,
		firstInputData.Height, firstOutputData.Units,
		convolutionConfig.FilterWidth(),
		convolutionConfig.FilterHeight(),
		firstInMemDesc.UnitOffset,
		firstInMemDesc.WidthOffset
		- convolutionConfig.FilterWidth() + 1,
		firstInMemDesc.HeightOffset
		- convolutionConfig.FilterHeight() + 1, 0, 0,
		firstInMemDesc.Width, firstInputData.Width,
		firstInMemDesc.Height, false));

		//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
		auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

		auto byteSize = convolutionConfig.FilterWidth()
		* convolutionConfig.FilterHeight()
		* convolutionConfig.FilterCount() * sizeof(T);
		if (maximumConstantBufferSize > byteSize)
		{
		kernel->SetConstantFilters(true);
		maximumConstantBufferSize -= byteSize;
		}

		byteSize = firstInMemDesc.TotalMemory() * sizeof(T);
		if (maximumConstantBufferSize > byteSize)
		{
		kernel->SetConstantInputDelta(true);
		maximumConstantBufferSize -= byteSize;
		}

		kernel->SetRelaxedMath(convolutionConfig.UseRelaxedMath());

		kernel->InitializeCompilerOptions();

		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
		this->context->AddKernel(kernel.get());

		kernel->SetFilters(filters.get());

		deviceAndBackConvolutionKernels.insert(make_pair(device, move(kernel)));
		}

		}

		template<class T>
		void ConvolutionLayer<T>::InitializeZeroKernel()
		{
		LayerDataDescription firstOutputData =
		this->outForwardPropDataDescriptions[0];
		LayerMemoryDescription firstInBackMemDesc =
		this->InBackPropMemoryDescriptions()[0];
		vector<OCLDevice*> devices = this->context->GetDevices();
		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

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

		unique_ptr<ZeroBorderKenel<T>> kernel(
		new ZeroBorderKenel<T>(firstOutputData.Width,
		firstOutputData.Height, firstOutputData.Units,
		borderStartLeft, borderStartRight, borderStartUp,
		borderStartDown, borderHorizontalSize,
		borderVerticalSize, firstInBackMemDesc.Width,
		firstInBackMemDesc.Height,
		firstInBackMemDesc.UnitOffset));

		kernel->SetUseRelaxedMath(convolutionConfig.UseRelaxedMath());
		kernel->InitializeCompilerOptions();

		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
		this->context->AddKernel(kernel.get());

		deviceAndZeroKernels.insert(make_pair(device, move(kernel)));
		}
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeMultiplyKernel()
		{
		LayerDataDescription firstOutputData =
		this->outForwardPropDataDescriptions[0];
		LayerMemoryDescription firstOutputBackMemDesc =
		this->OutBackPropMemoryDescriptions()[0];
		LayerMemoryDescription firstInBackMemDesc =
		this->InBackPropMemoryDescriptions()[0];
		LayerMemoryDescription firstInForwardMemDesc =
		this->InForwardPropMemoryDescriptions()[0];
		LayerDataDescription firstInputData =
		this->InForwardPropDataDescriptions()[0];

		vector<OCLDevice*> devices = this->context->GetDevices();
		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

		unique_ptr<MultiplyAllUnitsKernel<T>> kernel(
		new MultiplyAllUnitsKernel<T>(firstInputData.Width,
		firstInputData.Height, firstInputData.Units,
		firstInputData.Width, firstOutputBackMemDesc.Width,
		firstInForwardMemDesc.Width, 0, 0,
		firstOutputBackMemDesc.WidthOffset,
		firstOutputBackMemDesc.HeightOffset,
		firstOutputBackMemDesc.UnitOffset,
		firstInForwardMemDesc.WidthOffset,
		firstInForwardMemDesc.HeightOffset,
		firstInForwardMemDesc.UnitOffset,
		firstOutputBackMemDesc.Height,
		firstInForwardMemDesc.Height));

		auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

		auto deltaBytes = firstOutputBackMemDesc.TotalMemory() * sizeof(T);
		auto inputBytes = firstInForwardMemDesc.TotalMemory() * sizeof(T);

		if (maximumConstantBufferSize > deltaBytes)
		{
		kernel->SetUseConstantInputDelta(true);
		maximumConstantBufferSize -= deltaBytes;
		}

		if (maximumConstantBufferSize > inputBytes)
		{
		kernel->SetUseConstantInput(true);
		maximumConstantBufferSize -= inputBytes;
		}

		kernel->SetUseRelaxedMath(convolutionConfig.UseRelaxedMath());
		kernel->SetActivationFunction(this->BackPropActivationFunction());

		kernel->InitializeCompilerOptions();

		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
		this->context->AddKernel(kernel.get());

		deviceAndMultiplyKernels.insert(make_pair(device, move(kernel)));
		}
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeSumUnitKernel()
		{
		LayerMemoryDescription firstInBackMemDesc =
		this->InBackPropMemoryDescriptions()[0];
		LayerDataDescription firstOutputData =
		this->outForwardPropDataDescriptions[0];

		vector<OCLDevice*> devices = this->context->GetDevices();
		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

		//Since this is used for gradient calculation, we set offset to the entire
		//filter size
		unique_ptr<SumUnitKernel<T>> kernel(
		new SumUnitKernel<T>(firstInBackMemDesc.Width,
		firstInBackMemDesc.Height,
		firstInBackMemDesc.WidthOffset,
		firstInBackMemDesc.HeightOffset,
		firstInBackMemDesc.UnitOffset,
		convolutionConfig.FilterHeight()
		* convolutionConfig.FilterWidth() * convolutionConfig.FilterCount(), //Offset in the gradient
		convolutionConfig.FilterCount(), firstOutputData.Width,
		firstOutputData.Height));

		//HACK: this must be changed when removing deprecated functions
		unique_ptr<SumUnitKernel<T>> kernel2(
		new SumUnitKernel<T>(firstInBackMemDesc.Width,
		firstInBackMemDesc.Height,
		firstInBackMemDesc.WidthOffset,
		firstInBackMemDesc.HeightOffset,
		firstInBackMemDesc.UnitOffset, 0, //Zero in this case since we have splitted up the gradient in the new function
		convolutionConfig.FilterCount(), firstOutputData.Width,
		firstOutputData.Height));

		auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

		auto deltaBytes = firstInBackMemDesc.TotalMemory() * sizeof(T);

		if (maximumConstantBufferSize > deltaBytes)
		{
		kernel->SetConstantInput(true);
		kernel2->SetConstantInput(true);
		maximumConstantBufferSize -= deltaBytes;
		}

		kernel->SetRelaxedMath(convolutionConfig.UseRelaxedMath());
		kernel2->SetRelaxedMath(convolutionConfig.UseRelaxedMath());

		kernel->InitializeCompilerOptions();
		kernel2->InitializeCompilerOptions();

		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
		this->context->AddKernel(kernel.get());

		this->context->AddProgramFromSource(kernel2.get(), oneDeviceVector);
		this->context->AddKernel(kernel2.get());

		deviceAndSumUnitKernels.insert(make_pair(device, move(kernel)));
		deviceAndSumUnitKernels2.insert(make_pair(device, move(kernel2)));
		}
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeMultiplyWithOffsetKernel()
		{

		LayerMemoryDescription firstInBackMemDesc =
		this->InBackPropMemoryDescriptions()[0];
		LayerMemoryDescription firstInForwardMemDesc =
		this->InForwardPropMemoryDescriptions()[0];
		LayerDataDescription firstOutputData =
		this->outForwardPropDataDescriptions[0];
		LayerDataDescription firstInputData = this->InForwardPropDataDescriptions()[0];

		vector<OCLDevice*> devices = this->context->GetDevices();
		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

		//Since this is used for gradient calculation, we set offset to the entire
		//filter size
		unique_ptr<MultiplyWithOffsetKernel<T>> kernel(
		new MultiplyWithOffsetKernel<T>(convolutionConfig.FilterWidth(),
		convolutionConfig.FilterHeight(),
		convolutionConfig.FilterCount(), firstOutputData.Width,
		firstOutputData.Height, firstInBackMemDesc.Width,
		firstInBackMemDesc.Height,
		convolutionConfig.FilterWidth(),
		convolutionConfig.FilterHeight(),
		firstInputData.Width,
		0,
		0,
		firstInBackMemDesc.WidthOffset,
		firstInBackMemDesc.HeightOffset,
		firstInBackMemDesc.UnitOffset, 0, 0, 0));

		auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

		auto deltaBytes = firstInBackMemDesc.TotalMemory() * sizeof(T);

		if (maximumConstantBufferSize > deltaBytes)
		{
		kernel->SetConstantInputDelta(true);
		maximumConstantBufferSize -= deltaBytes;
		}


		auto inputBytes = firstInForwardMemDesc.TotalMemory() * sizeof(T);
		if (maximumConstantBufferSize > inputBytes)
		{
		kernel->SetConstantInput(true);
		maximumConstantBufferSize -= inputBytes;
		}


		kernel->SetRelaxedMath(convolutionConfig.UseRelaxedMath());

		kernel->InitializeCompilerOptions();

		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
		this->context->AddKernel(kernel.get());

		deviceAndMultiplyWithOffsetKernels.insert(make_pair(device, move(kernel)));
		}

		}

		*/

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
		void ConvolutionLayer<T>::EnqueueCalculateGradient(OCLDevice* device,
			int queueIndex, OCLMemory* previousInput, OCLMemory* delta,
			OCLMemory* gradient, bool blocking)
		{
			auto sumAllUnitsKernel = deviceAndSumKernels[device];
			sumAllUnitsKernel->SetMemoryArg(previousInput, 0);
			sumAllUnitsKernel->SetMemoryArg(summaryCache.get(), 1);
			auto multiplyWithOffsetKernel = deviceAndMultiplyWithOffsetKernels[device];
			multiplyWithOffsetKernel->SetMemoryArg(summaryCache.get(), 0);
			multiplyWithOffsetKernel->SetMemoryArg(delta, 1);
			multiplyWithOffsetKernel->SetMemoryArg(gradient, 2);
			auto sumUnitKernel = deviceAndSumUnitKernels[device];
			sumUnitKernel->SetMemoryArg(delta, 0);
			sumUnitKernel->SetMemoryArg(gradient, 1);

			device->ExecuteKernel(sumAllUnitsKernel, queueIndex, false);
			device->ExecuteKernel(multiplyWithOffsetKernel, queueIndex, false);
			device->ExecuteKernel(sumUnitKernel, queueIndex, blocking);
		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* delta, vector<OCLMemory*> gradient, bool blocking)
		{

			if (gradient.size() != 2)
				throw invalid_argument("The gradient size is not valid");

			if (gradient[0]->ByteSize() / sizeof(T) != (convolutionConfig.FilterCount() * convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight()))
				throw invalid_argument("The first gradient does not contain the correct amount of memory");

			if (gradient[1]->ByteSize() / sizeof(T) != (convolutionConfig.FilterCount()))
				throw invalid_argument("The second gradient does not contain the correct amount of memory");

			auto sumAllUnitsKernel = deviceAndSumKernels[device];
			sumAllUnitsKernel->SetMemoryArg(previousInput, 0);
			sumAllUnitsKernel->SetMemoryArg(summaryCache.get(), 1);
			auto multiplyWithOffsetKernel = deviceAndMultiplyWithOffsetKernels[device];
			multiplyWithOffsetKernel->SetMemoryArg(summaryCache.get(), 0);
			multiplyWithOffsetKernel->SetMemoryArg(delta, 1);
			multiplyWithOffsetKernel->SetMemoryArg(gradient[0], 2);
			auto sumUnitKernel = deviceAndSumUnitKernels2[device];
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
