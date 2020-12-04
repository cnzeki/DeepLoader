import MNN 
import numpy as np
import os
import time 
class MNNBackend():

    def init(self, model_path):
        assert os.path.exists(model_path), 'model_path not exist!'
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession({'numThread':4})

    def run(self, output_name, input_data):
        input_data = list(input_data.values())[0]
        input_tensor = self.interpreter.getSessionInput(self.session)
        ishape = input_tensor.getShape()
        self.interpreter.resizeTensor(input_tensor, input_data.shape)
        self.interpreter.resizeSession(self.session)
        ishape = input_tensor.getShape()
        tmp_input = MNN.Tensor(input_data.shape, MNN.Halide_Type_Float, input_data, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        out = self.interpreter.getSessionOutputAll(self.session)
        output = []
        for name, _ in out.items():
            shape = self.interpreter.getSessionOutput(self.session, name).getShape()
            out_data = self.interpreter.getSessionOutput(self.session, name).getData()
            out_data = np.reshape(out_data, shape)
            output.append(out_data)
        # out_data = np.expand_dims(np.reshape(out_data, (-1, 2)), axis=0)
        return output
