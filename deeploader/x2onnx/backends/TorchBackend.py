import torch

from .base import BaseBackend


class TorchBackend(BaseBackend):
    def init(self, torch_model):
        self.model = torch_model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_inputs(self):
        print('Can\'t infer input list for torch !')
        return []

    def get_outputs(self):
        print('Can\'t infer output list for torch !')
        return []

    def run(self, output_names, input_feed, run_options=None):
        with torch.no_grad():
            inputs = []
            for k, v in input_feed.items():
                # inputs.append(torch.from_numpy(v).cuda())
                inputs.append(torch.from_numpy(v).to(self.device))
            _outputs = self.model(*inputs)
            if isinstance(_outputs, torch.Tensor):
                _outputs = [_outputs]
            outputs = []
            for out in _outputs:
                outputs.append(out.cpu().detach().numpy())
            return outputs

    def close(self):
        self.model = None
