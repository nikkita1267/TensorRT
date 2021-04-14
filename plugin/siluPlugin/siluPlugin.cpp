#include "siluPlugin.h"

namespace {
const std::string plugin_name = "SiLU_TRT";
const std::string plugin_version = "001";
}

PluginFieldCollection SiLUPluginCreator::mFC{};
std::vector<PluginField> SiLUPluginCreator::mPluginAttributes;

SiLUPlugin::SiLUPlugin(const void* data, size_t length) {
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mBatchDim = read<int>(d);
    ASSERT(d == a + length)
}

int SiLUPlugin::getNbOutputs() const
{
    return 1;
}

Dims SiLUPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    nvinfer1::Dims outputDims{};
    outputDims.nbDims = inputs->nbDims;
    for (size_t i = 0; i < 8; ++i)
    {
        outputDims.d[i] = inputs->d[i];
    }
    return outputDims;
}

int SiLUPlugin::initialize()
{
    return STATUS_SUCCESS;
}

const char* SiLUPlugin::getPluginType() const
{
    return plugin_name.c_str();
}

const char* SiLUPlugin::getPluginVersion() const
{
    return plugin_version.c_str();
}

IPluginV2Ext* SiLUPlugin::clone() const
{
    auto* cloned = new SiLUPlugin();
    return cloned;
}

bool SiLUPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT || type == DataType::kINT32 || type == DataType::kINT8);
}

nvinfer1::DataType SiLUPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const
{
    return inputType[0];
}

void SiLUPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(nbInputs == 1)
    ASSERT(nbOutputs == 1)
    ASSERT(mBatchDim == 1)
    for (size_t i = 0; i < 8; ++i)
    {
        ASSERT(inputDims[0].d[i] == outputDims[0].d[i])
    }
    mDataType = inputTypes[0];
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

void SiLUPlugin::serialize(void* buffer) const {
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

size_t SiLUPlugin::getSerializationSize() const
{
    return sizeof(int);
}

void SiLUPlugin::destroy() {
    delete this;
}

void SiLUPlugin::setPluginNamespace(const char* pluginNamespace) {
    mPluginNamespace = pluginNamespace;
}

const char* SiLUPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

bool SiLUPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool SiLUPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

size_t SiLUPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

void SiLUPlugin::terminate() {}

const char* SiLUPluginCreator::getPluginName() const
{
    return plugin_name.c_str();
}

const char* SiLUPluginCreator::getPluginVersion() const
{
    return plugin_version.c_str();
}

const PluginFieldCollection* SiLUPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* SiLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    auto* plugin = new SiLUPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* SiLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    auto* plugin = new SiLUPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
