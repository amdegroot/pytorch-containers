# pytorch-containers

This repository aims to help former Torchies more seamlessly transition to the "Containerless" world of 
[PyTorch](https://github.com/pytorch/pytorch) 
by providing a list of PyTorch implementations of [Torch Table Layers](https://github.com/torch/nn/blob/master/doc/table.md).

### Table of Contents
- <a href='#concattable'>ConcatTable</a>
- <a href='#paralleltable'>ParallelTable</a>
- <a href='#maptable'>MapTable</a>
- <a href='#splittable'>SplitTable</a>
- <a href='#jointable'>JoinTable</a>
- <a href='#math-tables'>Math Tables</a>
- <a href='#intuitively-build-complex-architectures'>Intuitively Build Complex Architectures</a>


Note: As a result of full integration with [autograd](http://pytorch.org/docs/autograd.html), PyTorch requires networks to be defined in the following manner:
- Define all layers to be used in the `__init__` method of your network
- Combine them however you want in the `forward` method of your network (avoiding in place Tensor ops)

And that's all there is to it!

We will build upon a generic "TableModule" class that we initially define as:
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

input = torch.range(1,5)
net:forward(input)
```

### PyTorch
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
        self.layer1 = nn.Linear(5,5)
        self.layer2 = nn.Linear(5,10)
    def forward(self,x):
        y = [self.layer1(x),self.layer2(x)]
        return y
        
input = Variable(torch.range(1,5).unsqueeze(0))
net = TableModule()
net(input)
```

As you can see, PyTorch allows you to apply each member module that would have been
part of your Torch ConcatTable, directly to the same input Variable.  This offers much more 
flexibility as your architectures become more complex, and it's also a lot easier than 
remembering the exact functionality of ConcatTable, or any of the other tables for that matter.

Two other things to note: 
- To work with autograd, we must wrap our input in a `Variable` (we can also pass a python iterable of Variables)
- PyTorch requires us to add a batch dimension which is why we call `.unsqueeze(0)` on the input


## ParallelTable

### Torch
```Lua
net = nn.ParallelTable()
net:add(nn.Linear(10,5))
net:add(nn.Linear(5,10))

input1 = Torch.rand(1,10)
input2 = Torch.rand(1,5)
output = net:forward{input1,input2}
```

### PyTorch
```Python
class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
        self.layer1 = nn.Linear(10,5)
        self.layer2 = nn.Linear(5,10)
    def forward(self,x):
        y = [self.layer1(x[0]),self.layer2(x[1])]
        return y
        
input1 = Variable(torch.rand(1,10))
input2 = Variable(torch.rand(1,5))
net = TableModule()
output = net([input1,input2])
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
    def forward(self,x):
        y = [self.layer(member) for member in x]
        return y
        
input1 = Variable(torch.rand(1,5))
input2 = Variable(torch.rand(1,5))
input3 = Variable(torch.rand(1,5))
net = TableModule()
output = net([input1,input2,input3])
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
        y = x.chunk(x.size(dim),dim)
        return y
        
input = Variable(torch.rand(2,5))
net = TableModule()
output = net(input,1)
```
Alternatively, we could have used `torch.split()` instead of `torch.chunk()`. See the [docs](http://pytorch.org/docs/tensors.html).

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
        
    def forward(self,x,dim):
        y = torch.cat(x,dim)
        return y
        
input1 = Variable(torch.rand(1,5))
input2 = Variable(torch.rand(2,5))
input3 = Variable(torch.rand(3,5))
net = TableModule()
output = net([input1,input2,input3],0)
```
Note: We could have used torch.stack() instead of torch.cat(). See the [docs](http://pytorch.org/docs/tensors.html).

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
        x_sum = x1+x2 # could use sum() if input given as python iterable
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

The advantages that come with autograd when manipulating networks in these ways
become much more apparent with more complex architectures, so let's combine some of the 
operations we defined above. 

## Intuitively Build Complex Architectures 

Now we will visit a more complex example that combines several of the above operations.
The graph below is a random network that I created using the Torch [nngraph](https://github.com/torch/nngraph) package. The Torch model definition using nngraph can be found [here](https://github.com/amdegroot/pytorch-containers/blob/master/complex_graph.lua) and a raw Torch implementation can be found [here](https://github.com/amdegroot/pytorch-containers/blob/master/complex_net.lua) for comparison to the PyTorch code that follows. 

<img src= "https://github.com/amdegroot/pytorch-containers/blob/master/doc/complex_example.png" width="600px"/>

```Python
class Branch(nn.Module):
    def __init__(self,b2):
        super(Branch, self).__init__()
        """
        Upon closer examination of the structure, note a
        MaxPool2d with the same params is used in each branch, 
        so we can just reuse this and pass in the 
        conv layer that is repeated in parallel right after 
        it (reusing it as well).
        """
        self.b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b2 = b2
        
    def forward(self,x):
        x = self.b(x) 
        y = [self.b2(x).view(-1), self.b2(x).view(-1)] # pytorch 'ParallelTable'
        z = torch.cat((y[0],y[1])) # pytorch 'JoinTable'
        return z
```
Now that we have a branch class general enough to handle both branches, we can define the base segments 
and piece it all together in a very natural way. 

```Python
class ComplexNet(nn.Module):
    def __init__(self, m1, m2):
        super(ComplexNet, self).__init__()
        # define each piece of our network shown above
        self.net1 = m1 # segment 1 from VGG
        self.net2 = m2 #segment 2 from VGG
        self.net3 = nn.Conv2d(128,256,kernel_size=3,padding=1) # last layer 
        self.branch1 = Branch(nn.Conv2d(64,64,kernel_size=3,padding=1)) 
        self.branch2 = Branch(nn.Conv2d(128,256,kernel_size=3, padding=1))
         
    def forward(self, x):
        """
        Here we see that autograd allows us to safely reuse Variables in 
        defining the computational graph.  We could also reuse Modules or even 
        use loops or conditional statements.
        Note: Some of this could be condensed, but it is laid out the way it 
        is for clarity.
        """
        x = self.net1(x)
        x1 = self.branch1(x) # SplitTable (implicitly)
        y = self.net2(x) 
        x2 = self.branch2(y) # SplitTable (implicitly)
        x3 = self.net3(y).view(-1)
        output = torch.cat((x1,x2,x3),0) # JoinTable
        return output
```
This is a loop to define our VGG conv layers derived from [pytorch/vision](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py). (maybe a little overkill for our small case)
```Python
def make_layers(params, ch): 
    layers = []
    channels = ch
    for p in params:
            conv2d = nn.Conv2d(channels, p, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            channels = p
    return nn.Sequential(*layers) 
   
net = ComplexNet(make_layers([64,64],3),make_layers([128,128],64))
```
This documented python code can be found [here](https://github.com/amdegroot/pytorch-containers/blob/master/complex_net.py).  

