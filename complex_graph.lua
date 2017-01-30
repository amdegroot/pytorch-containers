require 'nn'
require 'nngraph'
--[[
This file contains the code definition of our complex example network using nngraph
in combination with some of the nn.Table modules that we defined.
]]--

input = nn.Identity()() -- input placeholder for nngraph

L1 = nn.SpatialConvolution(3,64,3,3,1,1,1,1)(input)
r1 = nn.ReLU()(L1)
L2 = nn.SpatialConvolution(64,64,3,3,1,1,1,1)(r1)
r2 = nn.ReLU()(L2)
L3 = nn.SpatialConvolution(64,128,3,3,1,1,1,1)(r2)
r3 = nn.ReLU()(L3)
L4 = nn.SpatialConvolution(128,128,3,3,1,1,1,1)(r3)
r4 = nn.ReLU()(L4)
r5 = nn.View(-1)(nn.SpatialConvolution(128,256,3,3,1,1,1,1)(r4))

b2 = nn.SpatialMaxPooling(2,2,2,2)(r4)
b2a = nn.SpatialConvolution(128,256,3,3,1,1,1,1) 
b2b = nn.SpatialConvolution(128,256,3,3,1,1,1,1)

leaf2a = nn.View(-1)(b2a(b2))
leaf2b = nn.View(-1)(b2b(b2))

leaf2 = nn.JoinTable(1)({leaf2a,leaf2b})

b1 = nn.SpatialMaxPooling(2,2,2,2)(r2)
b1a = nn.SpatialConvolution(64,64,3,3,1,1,1,1)
b1b = nn.SpatialConvolution(64,64,3,3,1,1,1,1)

leaf1a = nn.View(-1)(b1a(b1))
leaf1b = nn.View(-1)(b1b(b1))

leaf1 = nn.JoinTable(1)({leaf1a,leaf1b})

output = nn.JoinTable(1)({leaf1,r5,leaf2})

net = nn.gModule({input},{output})
out = net:forward(torch.Tensor(1,3,10,10))
graph.dot(net.fg, 'Forward Graph', 'complexgraph')
graph.dot(net.bg, 'Backward Graph','bcomplexgraph')