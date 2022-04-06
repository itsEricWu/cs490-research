from torchvision import models
from torchsummary import summary

from resenet_model import Net

model = Net(l=512)
summary(model, (3, 48, 48))

model = models.resnet18()
summary(model, (3, 48, 48))
