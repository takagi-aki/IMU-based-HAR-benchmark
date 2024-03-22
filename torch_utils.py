import os

import torch
from torch import nn

class TrialModel(nn.Module):
    def __init__(self, mod, out_loss, out_activ, **args) -> None:
        super().__init__()
        self.mod = mod
        self.out_activ = out_activ
        self.out_loss = out_loss

    def on_fit_start(self, class_weight, learning_rate, **args) -> torch.optim.Optimizer:
        # NOTE: setting hyperplameters here
        #       ex) class weight, learning_rate, activation_func...
        #       return torch.optim.Optimizer
        raise NotImplementedError(f'TrialModel is interfce. Define TrialModel child and implement code.')

    def on_train_batch(self, x, y) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: Implement behavior on training.
        #       Then, return prediction and loss.
        raise NotImplementedError(f'TrialModel is interfce. Define TrialModel child and implement code.')


    def on_test_batch(self, x, y) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: Implement behavior on test/validation.
        #       Then, return prediction and loss.
        raise NotImplementedError(f'TrialModel is interfce. Define TrialModel child and implement code.')


class BasicTrialModel(TrialModel):
    def __init__(self, mod, out_loss, out_activ) -> None:
        super().__init__(mod, out_loss, out_activ)

    def on_fit_start(self, class_weight, learning_rate, **args):
        if self.out_activ == 'softmax':
            self.activfunc = nn.Softmax()
        else:
            raise NotImplementedError(f'This activation func option [{self.out_activ}] is not implemented by pytorch mode.')

        if self.out_loss == 'categorical_crossentropy':
            if(class_weight is not None):
                _w = torch.Tensor(class_weight)
            else:
                _w = None
            self.lossfunc_train:torch.nn.BCELoss = torch.nn.BCELoss(weight=_w)
            self.lossfunc_test:torch.nn.BCELoss = torch.nn.BCELoss(weight=None)
        else:
            raise NotImplementedError(f'This loss func option [{self.out_loss}] is not implemented by pytorch mode.')
        
        return torch.optim.Adam(self.mod.parameters(), lr=learning_rate)

    def on_train_batch(self, x, y):
        pred = self.activfunc(self.mod(x))
        loss = self.lossfunc_train(pred, y)
        return pred, loss

    def on_test_batch(self, x, y):
        pred = self.activfunc(self.mod(x))
        loss = self.lossfunc_test(pred, y)
        return pred, loss


def save_model(model: nn.Module, path: str):
    model.cpu()
    _save_model_dir = os.path.dirname(path)
    if(not os.path.exists(_save_model_dir)):
        os.mkdir(_save_model_dir)
    torch.save(model.state_dict(), path)
