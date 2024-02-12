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

#First Modify the data path in the train_MMNet_V1.py
#Use training data: [UofSC_Train_V4](https://www.dropbox.com/scl/fo/06t1fw89s07kk116y955a/h?rlkey=s02t05nlzupoy3548xjhldn6p&dl=0)

python3  train_MMNet_V1.py
####

#### Test the network:
#Use the checkpoint file and modify the checkpoint path in the test_MMNet_v1.py
#Use testing data: [UofSC_Test_V4](https://www.dropbox.com/scl/fo/l8avb3xj7popiu9x3qdhd/h?rlkey=bolpd1vhaoejor9ie9q6lnram&dl=0)

python3 test_MMNet_V1.py
####

#### Test the network on New Environment:
#Use the checkpoint file and modify the checkpoint path in the test_MMNet_v1_horizon.py
#Use testing data: [UofSC_Horizon](https://www.dropbox.com/scl/fo/tgiq9uscqdhgjzgtqtcvt/h?rlkey=0htb14yyg0qnk9gpwy8tscoui&dl=0)

python3 test_MMNet_V1_horizon.py
####
