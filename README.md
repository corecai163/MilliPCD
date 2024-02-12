## MilliPCD: Beyond Traditional Vision Indoor Point Cloud Generation via Handheld Millimeter-Wave Devices

#### Requirements
Nvidia GPU with CUDA support 
####

####
Install pytorch and torch-geometric via pip3 or conda.
####

####
Build EMD package
following readme.txt in the folder EMD

####

####
Check pip environment
using env_lib.txt
####


#### Train the network:

# First Modify the data path in the train_MMNet_V1.py
# Use training data: UofSC_Train_V4

python3  train_MMNet_V1.py
####

#### Test the network:
# Use the checkpoint file and modify the checkpoint path in the test_MMNet_v1.py
# Use testing data: UofSC_Test_V4

python3 test_MMNet_V1.py
####
