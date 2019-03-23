# multi-node multi-gpu MP example code

this code was tested in (1) single node with two GPUs and (2) two nodes with one GPU respectively

- Two nodes / each node has four GPUs /
- 4 RTX2080 : CUDA Version 10.0
- 4 GTX1080 : CUDA Version 9.0
- nccl is supported (CUDA version is >= 8.0)
- pytorch  1.0.1 / torchvision 0.2.1
- every node has authorized_keys of each other
- because this code uses "send" and "recv" function in distributed module so this code uses "Gloo" instead of "Nccl"
- Please don't forget to set environment variable for GLOO_SOCKET_IFTNAME

    $ export GLOO_SOCKET_IFNAME=eth0


#### How to run

    Run in multi-node which have one GPU respectivley.
    $ python multinodeMP.py  --IP={MASTER IP} --portNum={Master PortNum} --GPU-list=1,1 --node-index={Node index 0 or 1} --batch-size=256

    please change --node-index for each node

    Run in single node with two GPUs
    $ python multinodeMP.py  --IP={MASTER IP} --portNum={Master PortNum} --GPU-list=2 --node-index=0 --batch-size=256

