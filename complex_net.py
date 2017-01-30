import torch
import torch.nn as nn
from torch.autograd import Variable


class Branch(nn.Module):
    def __init__(self,b2):
        """Constructs each branch necessary depending on input
    
        Args: 
            b2(nn.Module()): An nn.Conv2d() is passed with specific params
      
        """
        super(Branch, self).__init__()
        self.b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b2 = b2
        
    def forward(self,x):
        x = self.b(x) 
        y = [self.b2(x).view(-1), self.b2(x).view(-1)]
        z = torch.cat((y[0],y[1]))
        return z
        
        
class ComplexNet(nn.Module):
    def __init__(self, m1, m2):
        """Constructs the base of the network and attaches the branches

        Args: 
            m1(nn.Sequential()): The first segment of the base network
            m2(nn.Sequential()): The second segment of the base network

        """
        super(ComplexNet, self).__init__()
        self.net1 = m1
        self.net2 = m2
        self.net3 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.branch1 = Branch(nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.branch2 = Branch(nn.Conv2d(128,256,kernel_size=3, padding=1))
         
    def forward(self, x):
        x = self.net1(x)
        x1 = self.branch1(x)
        y = self.net2(x)
        x2 = self.branch2(y)
        x3 = self.net3(y).view(-1)
        output = torch.cat((x1,x2,x3),0)
        return output
    
def make_layers(params, ch): 
    """Constructs the base segments of the network and returns as nn.Sequential()

    Args: 
        params(int[]): The Conv2d parameters for input and output channels
        ch(int): The initial input channel parameter for the first Conv2d

    """

    layers = []
    channels = ch
    for p in params:
            conv2d = nn.Conv2d(channels, p, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            channels = p
    return nn.Sequential(*layers) 
   
net = ComplexNet(make_layers([64,64],3),make_layers([128,128],64))
return net