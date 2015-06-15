/*
 * StandardOutputLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "StandardOutputLayer.h"
#include "Matuna.ConvNet/InterlockHelper.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"
#include "Matuna.Helper/Converter.h"

#include <stdexcept>
#include <type_traits>
#include <string>

namespace Matuna
{
namespace MachineLearning
{

template<class T>
StandardOutputLayer<T>::StandardOutputLayer(shared_ptr<OCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		MatunaActivationFunction backPropActivation,
		const StandardOutputLayerConfig* outputLayerConfig) :
		OutputLayer(inputLayerDescriptions, backPropActivation,
				outputLayerConfig), context(context), config(*outputLayerConfig)
{

	if (inputLayerDescriptions.size() == 0)
		throw invalid_argument(
				"There's no input data descriptions for the standard output layer.");

	//TODO: Make sure this works for a more general case later on.
	if (inputLayerDescriptions.size() > 1)
	{
		auto count = inputLayerDescriptions.size();
		for (int i = 1; i < count; i++)
			if (!InterlockHelper::DataEquals(inputLayerDescriptions[i - 1],
					inputLayerDescriptions[i]))
				throw invalid_argument(
						"We cannot have multiple different input descriptions for a standard output layer");
	}

	//The targets must have the same data descriptions as the inputs
	inBackPropDataDescriptions = inputLayerDescriptions;
	inputDescription = inputLayerDescriptions[0];

	for (auto& inputDescription : inputLayerDescriptions)
	{
		LayerMemoryDescription inBackPropMemProp;
		inBackPropMemProp.Height = inputDescription.Height;
		inBackPropMemProp.Width = inputDescription.Width;
		inBackPropMemProp.Units = inputDescription.Units;
		inBackPropMemProp.UnitOffset = 0;
		inBackPropMemProp.WidthOffset = 0;
		inBackPropMemProp.HeightOffset = 0;
		inBackPropMemoryProposals.push_back(inBackPropMemProp);
		outBackPropMemoryProposals.push_back(inBackPropMemProp);
		inForwardPropMemoryProposals.push_back(inBackPropMemProp);
	}

	useImage = false;
}

template<class T>
StandardOutputLayer<T>::~StandardOutputLayer()
{

}

template<class T>
void StandardOutputLayer<T>::InterlockFinalized()
{

	auto inBackProp = inBackPropDataDescriptions[0];
	auto inBackPropMem = this->InBackPropMemoryDescriptions()[0];
	auto inForwardPropMem = this->InForwardPropMemoryDescriptions()[0];
	auto outBackPropMem = this->OutBackPropMemoryDescriptions()[0];

	if (!InterlockHelper::DataEquals(inputDescription, inBackProp))
		throw runtime_error("The targets are not the same as the inputs");

	if (!InterlockHelper::MemoryEquals(inBackPropMem, inForwardPropMem))
		throw runtime_error(
				"The inBackProp memory and the inForwardProp memory doesn't correspond");

	//Make sure the type we want to execute is supported on the device.
	vector<OCLDevice*> devices = this->context->GetDevices();
	for (auto device : devices)
	{
		auto deviceInfo = device->DeviceInfo();
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
	}

	unordered_map<OCLDevice*, unique_ptr<OCLProgram>> programs;
	auto activationFunction = this->BackPropActivationFunction();
	for (auto device : devices)
	{
		unique_ptr<OCLProgram> program(new OCLProgram());

		program->SetUseRelaxedMath(config.UseRelaxedMath());
		if (config.ComputationPrecision() == MatunaHalfPrecision)
			program->AddDefine("HALF_MATH");
		else if (config.ComputationPrecision() == MatunaNativePrecision)
			program->AddDefine("NATIVE_MATH");

		program->AddIncludePath(OCLProgram::DefaultSourceLocation);

		if (is_same<cl_double, T>::value)
			program->AddDefine("DOUBLE_PRECISION");

		if (activationFunction == MatunaSigmoidActivation)
			program->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_SIGMOID");
		else if (activationFunction == MatunaTanhActivation)
			program->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_TANH");

		program->SetName(
				"OutputLayerProgram"
						+ Converter::ConvertToString(program->InstanceCount()));

		programs.insert(make_pair(device, move(program)));
	}

	if (outBackPropMem.Width != 1 || outBackPropMem.Height != 1)
	{
		InitializeImageProgram(programs);
		useImage = true;
	}
	else
		InitializeProgram(programs);

	//Attaching and compiling all the programs
	for (auto device : devices)
	{
		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		context->AttachProgram(move(programs[device]), oneDeviceVector);
	}
}

template<class T>
void StandardOutputLayer<T>::InitializeImageProgram(
		unordered_map<OCLDevice*, unique_ptr<OCLProgram>>& programs)
{
	auto inForwardPropMem = this->InForwardPropMemoryDescriptions()[0];
	auto inForwardPropData = this->InForwardPropDataDescriptions()[0];
	auto outBackPropMem = this->OutBackPropMemoryDescriptions()[0];

	string outputSourcePath = Path::Combine(OCLProgram::DefaultSourceLocation,
			"ImageOutputBackProp.cl");
	string errorSourcePath = Path::Combine(OCLProgram::DefaultSourceLocation,
			"ImageOutputError.cl");
	string includePath = OCLProgram::DefaultSourceLocation;
	auto errorFunction = config.ErrorFunction();

	int globalUnits = inForwardPropData.Units;
	int globalWidth = inForwardPropData.Width;
	int globalHeight = inForwardPropData.Height;

	int inputUnitMemoryWidthOffset = inForwardPropMem.WidthOffset;
	int inputUnitMemoryHeightOffset = inForwardPropMem.HeightOffset;
	int inputUnitOffset = inForwardPropMem.UnitOffset;

	int outputUnitMemoryWidthOffset = outBackPropMem.WidthOffset;
	int outputUnitMemoryHeightOffset = outBackPropMem.HeightOffset;
	int outputUnitOffset = outBackPropMem.UnitOffset;

	int inputUnitMemoryWidth = inForwardPropMem.Width;
	int inputUnitMemoryHeight = inForwardPropMem.Height;

	int outputUnitMemoryWidth = outBackPropMem.Width;
	int outputUnitMemoryHeight = outBackPropMem.Height;

	vector<OCLDevice*> devices = this->context->GetDevices();
	for (auto device : devices)
	{
		auto deviceInfo = device->DeviceInfo();
		auto maximumConstantBufferSizeOriginal =
				deviceInfo.MaxConstantBufferSize();

		unique_ptr<LayerKernel<T>> imageOutputKernel(new LayerKernel<T>());
		auto maximumConstantBufferSize = maximumConstantBufferSizeOriginal;
		auto inputTargetBytes = sizeof(T) * inForwardPropMem.TotalMemory();

		imageOutputKernel->AddSourcePath(outputSourcePath);
		imageOutputKernel->AddIncludePath(includePath);

		imageOutputKernel->SetKernelName("BackPropagation");

		if (maximumConstantBufferSize > inputTargetBytes)
		{
			imageOutputKernel->AddDefine(outputSourcePath, "CONSTANT_INPUT");
			maximumConstantBufferSize -= inputTargetBytes;
		}
		if (maximumConstantBufferSize > inputTargetBytes)
		{
			imageOutputKernel->AddDefine(outputSourcePath, "CONSTANT_TARGET");
			maximumConstantBufferSize -= inputTargetBytes;
		}

		imageOutputKernel->AddGlobalSize(globalWidth);
		imageOutputKernel->AddGlobalSize(globalHeight);
		imageOutputKernel->AddGlobalSize(globalUnits);

		imageOutputKernel->AddDefineSubsitute(outputSourcePath,
				"INPUT_OFFSET_WIDTH", inputUnitMemoryWidthOffset);
		imageOutputKernel->AddDefineSubsitute(outputSourcePath,
				"INPUT_OFFSET_HEIGHT", inputUnitMemoryHeightOffset);
		imageOutputKernel->AddDefineSubsitute(outputSourcePath,
				"INPUT_UNIT_OFFSET", inputUnitOffset);

		imageOutputKernel->AddDefineSubsitute(outputSourcePath,
				"OUTPUT_OFFSET_WIDTH", outputUnitMemoryWidthOffset);
		imageOutputKernel->AddDefineSubsitute(outputSourcePath,
				"OUTPUT_OFFSET_HEIGHT", outputUnitMemoryHeightOffset);
		imageOutputKernel->AddDefineSubsitute(outputSourcePath,
				"OUTPUT_UNIT_OFFSET", outputUnitOffset);

		imageOutputKernel->AddDefineSubsitute(outputSourcePath, "INPUT_STRIDE",
				inputUnitMemoryWidth);
		imageOutputKernel->AddDefineSubsitute(outputSourcePath, "OUTPUT_STRIDE",
				outputUnitMemoryWidth);

		imageOutputKernel->AddDefineSubsitute(outputSourcePath,
				"OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING",
				outputUnitMemoryHeight * outputUnitMemoryWidth);
		imageOutputKernel->AddDefineSubsitute(outputSourcePath,
				"INPUT_UNIT_ELEMENT_COUNT_INC_PADDING",
				inputUnitMemoryHeight * inputUnitMemoryWidth);

		bool useBinary = false;
		if (globalWidth == 1 && globalHeight == 1 && globalUnits == 1)
			useBinary = true;

		SetErrorFunctionDefine(imageOutputKernel.get(), outputSourcePath,
				useBinary);

		imageOutputKernels.insert(make_pair(device, imageOutputKernel.get()));
		programs[device]->AttachKernel(move(imageOutputKernel));
	}

	int dataWidth = inForwardPropData.Width;
	int dataHeight = inForwardPropData.Height;
	int dataUnits = inForwardPropData.Units;

	for (auto device : devices)
	{
		auto deviceInfo = device->DeviceInfo();
		auto maximumConstantBufferSizeOriginal =
				deviceInfo.MaxConstantBufferSize();

		unique_ptr<LayerKernel<T>> imageErrorKernel(new LayerKernel<T>());
		auto maximumConstantBufferSize = maximumConstantBufferSizeOriginal;
		auto inputTargetBytes = sizeof(T) * inForwardPropMem.TotalMemory();

		imageErrorKernel->SetKernelName("Error");
		imageErrorKernel->AddSourcePath(errorSourcePath);
		imageErrorKernel->AddIncludePath(includePath);

		if (maximumConstantBufferSize > inputTargetBytes)
		{
			imageErrorKernel->AddDefine(errorSourcePath, "CONSTANT_INPUT");
			maximumConstantBufferSize -= inputTargetBytes;
		}
		if (maximumConstantBufferSize > inputTargetBytes)
		{
			imageErrorKernel->AddDefine(errorSourcePath, "CONSTANT_TARGET");
			maximumConstantBufferSize -= inputTargetBytes;
		}

		//Refer to the notes for this
		if (errorFunction == MatunaMeanSquareError)
		{
			imageErrorKernel->AddDefine(errorSourcePath, "MSE");
		}
		else if (errorFunction == MatunaCrossEntropy)
		{
			if (dataUnits == 1 && dataWidth == 1 && dataHeight == 1)
				imageErrorKernel->AddDefine(errorSourcePath, "CE_BINARY");
			else
				imageErrorKernel->AddDefine(errorSourcePath, "CE");
		}
		else
			throw invalid_argument(
					"The error function is not supported by the output kernel");

		imageErrorKernel->AddDefineSubsitute(errorSourcePath,
				"INPUT_OFFSET_WIDTH", inputUnitMemoryWidthOffset);
		imageErrorKernel->AddDefineSubsitute(errorSourcePath,
				"INPUT_WIDTH_LIMIT",
				inputUnitMemoryWidthOffset + dataWidth);
		imageErrorKernel->AddDefineSubsitute(errorSourcePath,
				"INPUT_HEIGHT_LIMIT",
				inputUnitMemoryHeightOffset + dataHeight);
		imageErrorKernel->AddDefineSubsitute(errorSourcePath,
				"INPUT_OFFSET_HEIGHT", inputUnitMemoryHeightOffset);
		imageErrorKernel->AddDefineSubsitute(errorSourcePath,
				"INPUT_UNIT_OFFSET", inputUnitOffset);
		imageErrorKernel->AddDefineSubsitute(errorSourcePath, "INPUT_STRIDE",
				inputUnitMemoryWidth);
		imageErrorKernel->AddDefineSubsitute(errorSourcePath,
				"INPUT_UNIT_LIMIT", inputUnitOffset + dataUnits);
		imageErrorKernel->AddDefineSubsitute(errorSourcePath,
				"INPUT_UNIT_ELEMENT_COUNT_INC_PADDING",
				inputUnitMemoryWidth * inputUnitMemoryHeight);

		imageErrorKernels.insert(make_pair(device, imageErrorKernel.get()));
		programs[device]->AttachKernel(move(imageErrorKernel));
	}
}

template<class T>
void StandardOutputLayer<T>::SetErrorFunctionDefine(LayerKernel<T>* kernel,
		string path, bool binary)
{
	auto errorFunction = config.ErrorFunction();
	auto activationFunction = this->BackPropActivationFunction();
	//Refer to the notes for this
	if (errorFunction == MatunaMeanSquareError)
	{
		if (activationFunction == MatunaLinearActivation)
			kernel->AddDefine(path, "DIFFERENCE");
		else
			kernel->AddDefine(path, "MSE_ANY");
	}
	else if (errorFunction == MatunaCrossEntropy)
	{
		if (binary)
		{
			if (activationFunction == MatunaSigmoidActivation)
				kernel->AddDefine(path, "DIFFERENCE");
			else
				kernel->AddDefine(path, "CE_BINARY_ANY");
		}
		else
		{
			if (activationFunction == MatunaSoftMaxActivation)
				kernel->AddDefine(path, "DIFFERENCE");
			else
				kernel->AddDefine(path, "CE_ANY");
		}
	}
	else
		throw invalid_argument(
				"The error function is not supported by the output kernel");
}

template<class T>
void StandardOutputLayer<T>::InitializeProgram(
		unordered_map<OCLDevice*, unique_ptr<OCLProgram>>& programs)
{
	auto inForwardPropMem = this->InForwardPropMemoryDescriptions()[0];
	auto outBackPropMem = this->OutBackPropMemoryDescriptions()[0];
	//In the current implementation, I need to make sure the target (inBackProp)
	//and the (inForwardProp) are the same

	string outputSourcePath = Path::Combine(OCLProgram::DefaultSourceLocation,
			"OutputBackProp.cl");
	string errorSourcePath = Path::Combine(OCLProgram::DefaultSourceLocation,
			"OutputError.cl");
	string includePath = OCLProgram::DefaultSourceLocation;
	auto errorFunction = config.ErrorFunction();

	vector<OCLDevice*> devices = this->context->GetDevices();
	for (auto device : devices)
	{
		auto deviceInfo = device->DeviceInfo();
		auto maximumConstantBufferSizeOriginal =
				deviceInfo.MaxConstantBufferSize();

		unique_ptr<LayerKernel<T>> outputKernel(new LayerKernel<T>());
		auto maximumConstantBufferSize = maximumConstantBufferSizeOriginal;
		auto inputTargetBytes = sizeof(T) * inputDescription.Units;

		outputKernel->SetKernelName("BackPropagation");
		outputKernel->AddSourcePath(outputSourcePath);
		outputKernel->AddIncludePath(includePath);

		if (maximumConstantBufferSize > inputTargetBytes)
		{
			outputKernel->AddDefine(outputSourcePath, "CONSTANT_INPUT");
			maximumConstantBufferSize -= inputTargetBytes;
		}
		if (maximumConstantBufferSize > inputTargetBytes)
		{
			outputKernel->AddDefine(outputSourcePath, "CONSTANT_TARGET");
			maximumConstantBufferSize -= inputTargetBytes;
		}

		bool useBinary = inputDescription.Units == 1 ? true : false;
		SetErrorFunctionDefine(outputKernel.get(), outputSourcePath, useBinary);

		outputKernel->AddGlobalSize(inputDescription.Units);

		outputKernel->AddDefineSubsitute(outputSourcePath, "INPUT_UNIT_OFFSET",
				inForwardPropMem.UnitOffset);
		outputKernel->AddDefineSubsitute(outputSourcePath, "OUTPUT_UNIT_OFFSET",
				outBackPropMem.UnitOffset);

		outputKernels.insert(make_pair(device, outputKernel.get()));
		programs[device]->AttachKernel(move(outputKernel));
	}

	for (auto device : devices)
	{
		auto deviceInfo = device->DeviceInfo();
		auto maximumConstantBufferSizeOriginal =
				deviceInfo.MaxConstantBufferSize();

		unique_ptr<LayerKernel<T>> errorKernel(new LayerKernel<T>());
		auto maximumConstantBufferSize = maximumConstantBufferSizeOriginal;
		auto inputTargetBytes = sizeof(T) * inputDescription.Units;

		errorKernel->SetKernelName("Error");
		errorKernel->AddSourcePath(errorSourcePath);
		errorKernel->AddIncludePath(includePath);

		if (maximumConstantBufferSize > inputTargetBytes)
		{
			errorKernel->AddDefine(errorSourcePath, "CONSTANT_INPUT");
			maximumConstantBufferSize -= inputTargetBytes;
		}
		if (maximumConstantBufferSize > inputTargetBytes)
		{
			errorKernel->AddDefine(errorSourcePath, "CONSTANT_TARGET");
			maximumConstantBufferSize -= inputTargetBytes;
		}

		//Refer to the notes for this
		if (errorFunction == MatunaMeanSquareError)
		{
			errorKernel->AddDefine(errorSourcePath, "MSE");
		}
		else if (errorFunction == MatunaCrossEntropy)
		{
			if (inputDescription.Units == 1)
				errorKernel->AddDefine(errorSourcePath, "CE_BINARY");
			else
				errorKernel->AddDefine(errorSourcePath, "CE");
		}
		else
			throw invalid_argument(
					"The error function is not supported by the output kernel");

		errorKernel->AddDefineSubsitute(errorSourcePath, "INPUT_UNIT_OFFSET",
				inForwardPropMem.UnitOffset);
		errorKernel->AddDefineSubsitute(errorSourcePath, "INPUT_COUNT",
				inputDescription.Units);

		errorKernels.insert(make_pair(device, errorKernel.get()));
		programs[device]->AttachKernel(move(errorKernel));
	}
}

template<class T>
T StandardOutputLayer<T>::CalculateError(OCLDevice* device, int queueIndex,
		OCLMemory* previousInput, OCLMemory* target)
{

	T result;
	if (useImage)
	{
		auto& kernel = imageErrorKernels[device];
		kernel->SetMemoryArg(previousInput, 0);
		kernel->SetMemoryArg(target, 1);
		auto errorMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(T));
		kernel->SetMemoryArg(errorMemory.get(), 2);
		device->ExecuteTask(kernel, queueIndex, true);
		device->ReadMemory(errorMemory.get(), errorMemory->ByteSize(), &result,
				queueIndex, true);
	}
	else
	{
		auto& kernel = errorKernels[device];
		kernel->SetMemoryArg(previousInput, 0);
		kernel->SetMemoryArg(target, 1);
		auto errorMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(T));
		kernel->SetMemoryArg(errorMemory.get(), 2);
		device->ExecuteTask(kernel, queueIndex, true);
		device->ReadMemory(errorMemory.get(), errorMemory->ByteSize(), &result,
				queueIndex, true);
	}

	return result;
}

template<class T>
void StandardOutputLayer<T>::EnqueueBackPropagation(OCLDevice* device,
		int queueIndex, OCLMemory* previousInput, OCLMemory* target,
		OCLMemory* deltaOutput, bool blocking)
{
	if (useImage)
	{
		auto kernel = imageOutputKernels[device];
		kernel->SetMemoryArg(previousInput, 0);
		kernel->SetMemoryArg(target, 1);
		kernel->SetMemoryArg(deltaOutput, 2);
		device->ExecuteKernel(kernel, queueIndex, blocking);
	}
	else
	{
		auto& kernel = outputKernels[device];
		kernel->SetMemoryArg(previousInput, 0);
		kernel->SetMemoryArg(target, 1);
		kernel->SetMemoryArg(deltaOutput, 2);
		device->ExecuteKernel(kernel, queueIndex, blocking);
	}
}

template class StandardOutputLayer<cl_float> ;
template class StandardOutputLayer<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
