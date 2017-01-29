# pytorch-containers

This repository aims to help former Torchies more seemlessly transition to the "Containerless" world of 
[PyTorch](https://github.com/pytorch/pytorch) 
by providing a list of PyTorch implementations of [Torch Table Layers](https://github.com/torch/nn/blob/master/doc/table.md).

We will build our neural net from a simple nn.Sequential() module as this exists in both 
Torch and PyTorch and build up from there.  You will notice that as we add more and more  
complexity to our network, the Torch code becomes more and more verbose.  On the other hand,
thanks to autograd, the complexity of our PyTorch code does not increase at all.  


Note: As a result of full integration with Autograd, PyTorch requires networks to be defined in the following manner:
1. Define all layers to be used in the `__init__` method 
2. Combine them however you want in the `forward` method 

And that's all there is to it!

##ConcatTable

A simple Lua example:  
```Lua
net = nn.ConcatTable()
net:add(nn.Linear(5,5))
net:add(nn.Linear(5,10))

input = torch.range(1,5):view(1,5)
net:forward(input)
```

PyTorch Conversion: 
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(SimpleConcat,self).__init__()
        self.layer1 = nn.Linear(5,5).double()
        self.layer2 = nn.Linear(5,10).double()
    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        return x1,x2
        
input = Variable(torch.range(1,5).view(1,5))
net = SimpleConcat()
net(input)
```
As you can see, PyTorch allows you to apply each member module that would have been
part of your Torch ConcatTable, directly to the same input Variable.  This offers much more 
flexibility as your architectures become more complex.







