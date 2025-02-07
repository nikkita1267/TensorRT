#ifndef TENSORRT_SILUPLUGIN_H
#define TENSORRT_SILUPLUGIN_H

/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NvInferPlugin.h"
#include "kernel.h"
#include "plugin.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class SiLUPlugin : public IPluginV2Ext
{
public:
    SiLUPlugin() = default;

    SiLUPlugin(const void* data, size_t length);

    ~SiLUPlugin() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                         const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
                         const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

private:
    DataType mDataType;
    int mBatchDim;
    const char* mPluginNamespace;
    std::string mNamespace;
};

class SiLUPluginCreator : public BaseCreator
{
public:
    SiLUPluginCreator() = default;

    ~SiLUPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;

protected:
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_SILUPLUGIN_H
