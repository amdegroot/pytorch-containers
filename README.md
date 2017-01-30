# pytorch-containers

This repository aims to help former Torchies more seemlessly transition to the "Containerless" world of 
[PyTorch](https://github.com/pytorch/pytorch) 
by providing a list of PyTorch implementations of [Torch Table Layers](https://github.com/torch/nn/blob/master/doc/table.md).

### Table of Contents
- <a href='#concattable'>ConcatTable</a>
- <a href='#paralleltable'>ParallelTable</a>
- <a href='#maptable'>MapTable</a>
- <a href='#splittable'>SplitTable</a>
- <a href='#jointable'>JoinTable</a>
- <a href='#math-tables'>Math Tables</a>
- <a href='#build-more-complex-architectures'>Easily Build Complex Architectures</a>


Note: As a result of full integration with [autograd](http://pytorch.org/docs/autograd.html), PyTorch requires networks to be defined in the following manner:
1. Define all layers to be used in the `__init__` method of your network
2. Combine them however you want in the `forward` method of your network

And that's all there is to it!

We will build upon a generic "TableModule" class that we define as:
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
        self.layer1 = nn.Linear(5,5).double()
        self.layer2 = nn.Linear(5,10).double()
    def forward(self,x):
        ...
        ...
        ...
        return ...
```

## ConcatTable

### Torch
```Lua
net = nn.ConcatTable()
net:add(nn.Linear(5,5))
net:add(nn.Linear(5,10))

input = torch.range(1,5):view(1,5)
net:forward(input)
```

### PyTorch Conversion 
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
        self.layer1 = nn.Linear(5,5)
        self.layer2 = nn.Linear(5,10)
    def forward(self,x):
        y = [self.layer1(x),self.layer2(x)]
        return y
        
input = Variable(torch.range(1,5).view(1,5))
net = TableModule()
net(input)
```

As you can see, PyTorch allows you to apply each member module that would have been
part of your Torch ConcatTable, directly to the same input Variable.  This offers much more 
flexibility as your architectures become more complex.

Two other things to note: 
- To work with autograd, we must wrap our input in a Variable
- PyTorch requires us to add a batch dimension which is why we call `.view(1,5)` on the input


## ParallelTable

### Torch
```Lua
net = nn.ParallelTable()
net:add(nn.Linear(10,5))
net:add(nn.Linear(5,10))

input1 = Torch.range(1,10):view(1,10)
input2 = Torch.range(1,5):view(1,5)
output = net:forward{input1,input2}
```

### PyTorch
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
        self.layer1 = nn.Linear(10,5)
        self.layer2 = nn.Linear(5,10)
    def forward(self,x1,x2):
        y = [self.layer1(x1),self.layer2(x2)]
        return y
        
input1 = Variable(torch.range(1,10).view(1,10))
input2 = Variable(torch.range(1,5).view(1,5))
net = TableModule()
output = net(input1,input2)
```

## MapTable

### Torch
```Lua
net = nn.MapTable()
net:add(nn.Linear(5,10))

input1 = torch.rand(1,5)
input2 = torch.rand(1,5)
input3 = torch.rand(1,5)
output = net:forward{input1,input2,input3}
```

### PyTorch
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
        self.layer = nn.Linear(5,10)
    def forward(self,x1,x2,x3):
        y = [self.layer(x1),self.layer(x2),self.layer(x3)]
        return y
        
input1 = Variable(torch.rand(1,5))
input2 = Variable(torch.rand(1,5))
input3 = Variable(torch.rand(1,5))
net = TableModule()
output = net(input1,input2,input3)
```

## SplitTable

### Torch
```Lua
net = nn.SplitTable(2) # here we specify the dimension on which to split the input Tensor
input = torch.rand(2,5)
output = net:forward(input)
```

### PyTorch
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
    def forward(self,x,dim):
        return x.chunk(x.size(dim),dim)
        
input = Variable(torch.rand(2,5))
net = TableModule()
output = net(input,1)
```
Alternatively, we could have used torch.split() instead of torch.chunk(). See the [docs](http://pytorch.org/docs/tensors.html).

## JoinTable

### Torch
```Lua
net = nn.JoinTable(1)
input1 = torch.rand(1,5)
input2 = torch.rand(2,5)
input3 = torch.rand(3,5)
output = net:forward{input1,input2,input3}
```

### PyTorch
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
        
    def forward(self,x1,x2,x3,dim):
        return torch.cat((x1,x2,x3),dim)

input1 = Variable(torch.rand(1,5))
input2 = Variable(torch.rand(2,5))
input3 = Variable(torch.rand(3,5))
net = TableModule()
output = net(input1,input2,input3,0)
```
Note: We could have used torch.stack() instead of torch.cat(). See the [docs](http://pytorch.org/docs/tensors.html).

The advantages that come with autograd when manipulating networks in these ways
become much more apparent with more complex architectures, so let's combine some of the 
operations we defined above. 

## Math Tables

The math table implementations are pretty intuitive, so the Torch implementations are omitted in this repo, 
but just like the others, their well-written descriptions and examples can be found by visiting their official [docs](https://github.com/torch/nn/blob/master/doc/table.md).

### PyTorch Math
Here we define one class that executes all of the math operations. 
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
        
    def forward(self,x1,x2):
        x_sum = x1+x2
        x_sub = x1-x2
        x_div = x1/x2
        x_mul = x1*x2
        x_min = torch.min(x1,x2)
        x_max = torch.max(x1,x2)
        return x_sum, x_sub, x_div, x_mul, x_min, x_max

input1 = Variable(torch.range(1,5).view(1,5))
input2 = Variable(torch.range(6,10).view(1,5))
net = TableModule()
output = net(input1,input2)
print(output)
```
And we get: 
```
(Variable containing:
  7   9  11  13  15
[torch.FloatTensor of size 1x5]
, Variable containing:
-5 -5 -5 -5 -5
[torch.FloatTensor of size 1x5]
, Variable containing:
 0.1667  0.2857  0.3750  0.4444  0.5000
[torch.FloatTensor of size 1x5]
, Variable containing:
  6  14  24  36  50
[torch.FloatTensor of size 1x5]
, Variable containing:
 1  2  3  4  5
[torch.FloatTensor of size 1x5]
, Variable containing:
  6   7   8   9  10
[torch.FloatTensor of size 1x5]
)
```

## Building more complex architectures 

We will build our neural net from a base nn.Sequential() module, which exists in both 
Torch and PyTorch, and build up from there.  You will notice that as we add more and more  
complexity to our network, the Torch code becomes more and more verbose.  On the other hand,
thanks to autograd, the complexity of our PyTorch code does not increase at all. 
















