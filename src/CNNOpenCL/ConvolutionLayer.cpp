/*
* ConvolutionLayer.cpp
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#include "ConvolutionLayer.h"
#include <stdexcept>
#include <type_traits>
#include <random>

namespace ATML
{
	namespace MachineLearning
	{

		template<class T>
		ConvolutionLayer<T>::ConvolutionLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			ATMLActivationFunction backPropActivation,
			const ConvolutionLayerConfig* config) :
		OpenCLForwardBackPropLayer<T>(context, inputLayerDescriptions,
			backPropActivation, config), convolutionConfig(*config)
		{

			if (inputLayerDescriptions.size() == 0)
				throw invalid_argument(
				"There's no input data descriptions for the convolution layer.");

			if (inputLayerDescriptions.size() != 1)
				throw runtime_error("Not implemented exception");

			if (config->ConnectionType() != ATMLFullConnection)
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
			for (auto& deviceAndKernel : deviceAndConvolutionKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndSumKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndBackConvolutionKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndMultiplyKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndZeroKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndMultiplyWithOffsetKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndSumUnitKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}
		}

		template<class T>
		ConvolutionLayerConfig ConvolutionLayer<T>::GetConfig() const
		{
			return convolutionConfig;
		}

		template<class T>
		vector<Matrix<T>> ConvolutionLayer<T>::GetFilters() const
		{
			OpenCLDevice* device = this->context->GetDevices()[0];
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
			OpenCLDevice* device = this->context->GetDevices()[0];
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
			OpenCLDevice* device = this->context->GetDevices()[0];
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
			InitializeConvolutionKernel();
			InitializeSumAllKernel();
			InitializeBackConvolutionKernel();
			InitializeMultiplyKernel();
			InitializeZeroKernel();
			InitializeSumUnitKernel();
			InitializeMultiplyWithOffsetKernel();
		}

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

			vector<OpenCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

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

				vector<OpenCLDevice*> oneDeviceVector;
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

			vector<OpenCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

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
				vector<OpenCLDevice*> oneDeviceVector;
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

			vector<OpenCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

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

				vector<OpenCLDevice*> oneDeviceVector;
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
			vector<OpenCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

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

				vector<OpenCLDevice*> oneDeviceVector;
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

			vector<OpenCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

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

				vector<OpenCLDevice*> oneDeviceVector;
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

			vector<OpenCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

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

				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

				auto deltaBytes = firstInBackMemDesc.TotalMemory() * sizeof(T);

				if (maximumConstantBufferSize > deltaBytes)
				{
					kernel->SetConstantInput(true);
					maximumConstantBufferSize -= deltaBytes;
				}

				kernel->SetRelaxedMath(convolutionConfig.UseRelaxedMath());

				kernel->InitializeCompilerOptions();

				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());

				deviceAndSumUnitKernels.insert(make_pair(device, move(kernel)));
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

			vector<OpenCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

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

				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());

				deviceAndMultiplyWithOffsetKernels.insert(make_pair(device, move(kernel)));
			}

		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueForwardPropagation(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* output,
			bool blocking)
		{
			auto& sumAllUnitsKernel = deviceAndSumKernels[device];
			auto& convolutionKernel = deviceAndConvolutionKernels[device];
			sumAllUnitsKernel->SetInput(previousInput);
			sumAllUnitsKernel->SetOutput(summaryCache.get());
			device->ExecuteKernel(sumAllUnitsKernel.get(), queueIndex, false);
			convolutionKernel->SetInput(summaryCache.get());
			convolutionKernel->SetOutput(output);
			device->ExecuteKernel(convolutionKernel.get(), queueIndex, blocking);
		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueBackPropagation(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* deltaOutput, bool blocking)
		{
			auto& zeroKernel = deviceAndZeroKernels[device];
			zeroKernel->SetInputOutput(delta);
			auto& convolutionKernel = deviceAndBackConvolutionKernels[device];
			convolutionKernel->SetDeltaInput(delta);
			convolutionKernel->SetOutput(summaryCache.get());
			auto& multiplyKernel = deviceAndMultiplyKernels[device];
			multiplyKernel->SetInput(previousInput);
			multiplyKernel->SetInputDelta(summaryCache.get());
			multiplyKernel->SetOutput(deltaOutput);
			device->ExecuteKernel(zeroKernel.get(), queueIndex, false);
			device->ExecuteKernel(convolutionKernel.get(), queueIndex, false);
			device->ExecuteKernel(multiplyKernel.get(), queueIndex, blocking);
		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueCalculateGradient(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* gradient, bool blocking)
		{
			auto& sumAllUnitsKernel = deviceAndSumKernels[device];
			sumAllUnitsKernel->SetInput(previousInput);
			sumAllUnitsKernel->SetOutput(summaryCache.get());
			auto& multiplyWithOffsetKernel = deviceAndMultiplyWithOffsetKernels[device];
			multiplyWithOffsetKernel->SetInput(summaryCache.get());
			multiplyWithOffsetKernel->SetInputDelta(delta);
			multiplyWithOffsetKernel->SetOutput(gradient);
			auto& sumUnitKernel = deviceAndSumUnitKernels[device];
			sumUnitKernel->SetInput(delta);
			sumUnitKernel->SetOutput(gradient);

			device->ExecuteKernel(sumAllUnitsKernel.get(), queueIndex, false);
			device->ExecuteKernel(multiplyWithOffsetKernel.get(), queueIndex, false);
			device->ExecuteKernel(sumUnitKernel.get(), queueIndex, blocking);
		}

		template<class T>
		vector<tuple<OpenCLMemory*, int>> ConvolutionLayer<T>::GetParameters()
		{
			vector<tuple<OpenCLMemory*, int> > result;

			auto filterTuple = make_tuple(filters.get(),
				convolutionConfig.FilterCount() * convolutionConfig.FilterWidth()
				* convolutionConfig.FilterHeight());
			auto biasTuple = make_tuple(biases.get(), convolutionConfig.FilterCount());
			result.push_back(filterTuple);
			result.push_back(biasTuple);

			return result;
		}

		template<class T>
		void ConvolutionLayer<T>::GetParameters(T* parameters, OpenCLDevice* device,
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
		void ConvolutionLayer<T>::SetParameters(T* parameters, OpenCLDevice* device,
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
} /* namespace ATML */
