import torch
import sys
sys.path.append("/home/lu677/cs490/cs490-research/Resnet")


def test():
    print(torch.cuda.is_available())


test()
