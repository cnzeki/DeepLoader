#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

import random
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRTBackend():
    def GiB(self, val):
        return val * 1 << 30
    def init(self, onnx_file, ModelData, output_shape):
        self.EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.ModelData = ModelData
        self.output_shape = output_shape
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.engine = self.get_engine(onnx_file)
        #self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(engine)
        # self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine, self.input_shape[0][0])
        #self.inputs, self.outputs, self.bindings, self.stream=None,None,None,None
        self.context = self.engine.create_execution_context()
        # self.ctx = cuda.Device(0).make_context()

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    def allocate_buffers(self, engine, batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            bind = engine.get_binding_shape(binding)
            vol = trt.volume(bind)
            # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            if size < 0:
                size *= -1
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference_v2(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def get_engine(self, onnx_file_path):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        def build_engine():
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network(
                    self.EXPLICIT_BATCH) as network, \
                    trt.OnnxParser(network, self.TRT_LOGGER) as parser,\
                    builder.create_builder_config() as config:
                builder.max_batch_size = 256
                # Parse model file
                # builder.fp16_mode = True
                # builder.strict_type_constraints = True


                # config.set_flag(trt.BuilderFlag.FP16)
                config.max_workspace_size=self.GiB(2)
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                        onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    if not parser.parse(model.read()):
                        print('ERROR: Failed to parse the ONNX file.')
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None

                # network.get_input(0).shape = self.input_shape[0]
                # network.get_input(0).shape = [1, 3, 112, 112]
                # print('Completed parsing of ONNX file')
                # print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

                ############################################
                #config=builder.create_builder_config()
                if self.ModelData:
                    profile = builder.create_optimization_profile()
                    profile.set_shape(network.get_input(0).name, self.ModelData[0], self.ModelData[1], self.ModelData[2])
                    config.add_optimization_profile(profile)
                    engine = builder.build_engine(network,config)
                else:
                    engine = builder.build_cuda_engine(network)

                ##################################################

                # engine = builder.build_cuda_engine(network)
                # print("Completed creating Engine")
                # with open(engine_file_path, "wb") as f:
                #     f.write(engine.serialize())
                return engine

        # if os.path.exists(engine_file_path):
        #     # If a serialized engine exists, use it instead of building an engine.
        #     print("Reading engine from file {}".format(engine_file_path))
        #     with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
        #         return runtime.deserialize_cuda_engine(f.read())
        # else:

        return build_engine()
    def run(self, output_name, data):
        # self.ctx.push()
        data = list(data.values())[0]

        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, data.shape[0])
        self.context.set_binding_shape(0, data.shape)
        
        np.copyto(self.inputs[0].host.reshape(data.shape), data)
        # np.copyto(self.inputs[0].host, data)

        trt_outputs = self.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        out = []
        for i in range(len(trt_outputs)):
            self.output_shape[i][0] = data.shape[0]
            out.append(trt_outputs[i].reshape(self.output_shape[i]))

        # self.ctx.pop()
        # s1 =  self.context.get_binding_shape(0)
        # s2 =  self.context.get_binding_shape(1)
        return out
if __name__ == '__main__':
    # main()
    import sys 
    proj_dir = '/home/ysten/chenbo/video-engine'
    alg_root = os.path.join(proj_dir, 'alg')
    sys.path.append(os.path.join(alg_root, 'video_pred'))
    sys.path.append(proj_dir)
    sys.path.append(alg_root)
    import easydict
    import torch 
    model_file = 'models/videopred/pytorch/resnetrgb16.pth'
    onnx_model_file = 'models/videopred/onnx/rgb_resnet50.onnx'
    from deeploader.x2onnx.backends.TorchBackend import TorchBackend

    load = torch.load(model_file, map_location='cuda')
    if isinstance(load, torch.nn.DataParallel):
        model = load.module
    else:
        model = load
    sess = TorchBackend()
    sess.init(model)

    data = np.random.rand(1,16, 3, 224, 224)
    trt_engine = TensorRTBackend()
    trt_engine.init(onnx_model_file, [[1, 16 , 3, 224, 224], [2, 16 , 3, 224, 224], [4, 16 , 3, 224, 224]],  [['?', 8]])

    # trt_engine = TensorRTBackend()
    # trt_engine.init('models/videopred/onnx/flow_resnet50.onnx', (2, 16, 3, 224, 224), (2, 8))
    import time
    trt_time = time.time()
    times = 50
    for i in range(times):
        out1 = trt_engine.run([], {'input':data})
    print('trt_usetime:', time.time()-trt_time)
    rt_time = time.time()
    for i in range(times):
        out2 = sess.run([], {'input':data.astype(np.float32)})
        
    print('rt_usetime:', time.time()-rt_time)
    for i in range(len(out1)):
        np.testing.assert_almost_equal(out1[i], out2[i], decimal=2, verbose=True)
        print('done:{}'.format(i))
    print('equal!!!!!!!')
    print(out1)
    print(out2)
    

