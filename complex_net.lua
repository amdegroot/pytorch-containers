require 'nn'

--[[
This file contains the code definition of our complex example network in raw Torch
using nn.Table modules.  Since nngraph is not used, this network is defined in
reverse order in terms of sequential segments and branches.
]]--


-- the final layer of our base network 
net3 = nn.Sequential()
net3:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))
net3:add(nn.View(-1))

-- split the last branch 
branch2b = nn.ParallelTable()
branch2b:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))
branch2b:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))

-- here we add max pooling and use nn.Replicate() --> nn.SplitTable
-- to allow for a ParallelTable to split the branch 
branch2a = nn.Sequential()
branch2a:add(nn.SpatialMaxPooling(2,2,2,2))
branch2a:add(nn.Replicate(4))
branch2a:add(nn.SplitTable(1))
branch2a:add(branch2b)

-- second (final) branch from base of network
branch2 = nn.ParallelTable()
branch2:add(branch2a)
branch2:add(net3)

-- using nn.Sequential() for part 2 of base network 
net2 = nn.Sequential() 
net2:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1))
net2:add(nn.ReLU())
net2:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1))
net2:add(nn.ReLU())
net2:add(nn.Replicate(4))
net2:add(nn.SplitTable(1))
net2:add(branch2)

-- split from first branch 
branch1b = nn.ParallelTable()
branch1b:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
branch1b:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))

-- add a max pooling layer to branch and then prepare to split 
branch1a = nn.Sequential()
branch1a:add(nn.SpatialMaxPooling(2,2,2,2))
branch1a:add(nn.Replicate(4))
branch1a:add(nn.SplitTable(1))
branch1a:add(branch1b)

-- first branch from the base of the network
branch1 = nn.ParallelTable()
branch1:add(branch1a)
branch1:add(net2)

-- the first layers of the network 
net = nn.Sequential()
net:add(nn.SpatialConvolution(3,64,3,3,1,1,1,1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.ReLU())
net:add(nn.Replicate(4))
net:add(nn.SplitTable(1))
net:add(branch1)
