from dataset import *


d = get_cifar_10("F:\\dataset\\cifar", "train")

print(len(d))
print(d[0], d[-1])

d = get_cifar_100("F:\\dataset\\cifar", "train")

print(len(d))
print(d[0], d[-1])

d = get_imagenet("F:\\dataset\\ILSVRC2012", "train", 5)
print(len(d))
print(d[0], d[-1])
