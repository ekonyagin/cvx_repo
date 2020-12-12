import torch
import torch.nn as nn
import torchvision
import ddn.pytorch.robustpool as robustpool


class RobustPoolSqueezeNet(nn.Module):
    def __init__(self, num_classes, robust_type, alpha=1.0):
        super().__init__()
        
        if robust_type != "vanilla":
            features = torchvision.models.squeezenet1_0(pretrained=False, num_classes=num_classes).features
            classifier = nn.Sequential(
                nn.Linear(512, 100),
                nn.ReLU(),
                nn.Linear(100, num_classes))
            
            self._model_name = "squeezenet_1_0_"

            if robust_type == "quadratic":
                pooling = robustpool.RobustGlobalPool2d(robustpool.Quadratic, alpha=alpha)
                self._model_name += "quadratic" # + str(alpha).replace('.', '_')
            elif robust_type == "huber":
                pooling = robustpool.RobustGlobalPool2d(robustpool.Huber, alpha=alpha)
                self._model_name += "huber"
            elif robust_type == "pseudo-huber":
                pooling = robustpool.RobustGlobalPool2d(robustpool.PseudoHuber, alpha=alpha)
                self._model_name += "pseudo_huber"
            elif robust_type == "welsch":
                pooling = robustpool.RobustGlobalPool2d(robustpool.Welsch, alpha=alpha)
                self._model_name += "welsch"
            elif robust_type == "trunc-quadratic":
                pooling = robustpool.RobustGlobalPool2d(robustpool.TruncatedQuadratic, alpha=alpha)
                self._model_name += "trunc_quadratic"

            self._model_name += "_alpha_" + str(alpha).replace('.', '_')

            self._model = nn.Sequential(features, pooling, classifier)

        else:
            self._model = torchvision.models.squeezenet1_0(pretrained=False, num_classes=num_classes)
            self._model_name = "squeezenet_1_0_vanilla"
    
    def forward(self, x):
        return self._model.forward(x)

    def get_model_name(self):
        return self._model_name

