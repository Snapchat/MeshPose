import torch

from meshpose.architecture.backbones import Hrnet32
from meshpose.architecture.modules import EncDecModel, SequentialModel, Model, MultiBranchNet


class ModelScheduler(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleDict(models)

    def load_model_state_dict(self, state_dict, strict=True):

        for key, module in self.models.items():
            model_state_dict = state_dict[key]
            module.load_state_dict(model_state_dict, strict=strict)

    def forward(self, input_crop):
        return self.models["root"](input_crop)


def load_model(checkpoint):

    hrnet = Hrnet32()
    encdec = EncDecModel(encoder=hrnet)
    task_heads = MultiBranchNet()
    sequential = SequentialModel(modules=[encdec, task_heads])
    root_model = Model(model=sequential)
    models = {"root": root_model}
    main_model = ModelScheduler(models)
    saved_data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    main_model.load_model_state_dict(saved_data["model"], strict=False)
    main_model.eval()
    return main_model


class MeshPoseModel(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.main_model = load_model(checkpoint)

    def forward(self, images):
        with torch.no_grad():
            outputs = self.main_model(images)
        return outputs
