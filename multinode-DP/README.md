# multi-node multi-gpu DP example code

this code was tested in condition described below.

- Two nodes / each node has four GPUs /
- 4 RTX2080 : CUDA Version 10.0
- 4 GTX1080 : CUDA Version 9.0
- nccl is supported (CUDA version is >= 8.0)
- pytorch  1.0.1 / torchvision 0.2.1
- every node has authorized_keys of each other


#### How to run

    $ python multinodeDP.py --dist-url=tcp://MASTER_IP:MASTER_PORT --backend=nccl --node-index=0 --GPU-list=4,4

please change --node-index for each node
