import torch
import os

def dist_init(host_addr, rank, local_rank, world_size, port=2768):
    print("###Distributed Initialized###")
    print("host_addr:", host_addr)
    print("rank:", rank)
    print("local_rank:", local_rank)
    print("world_size:", world_size)
    print("port:", port)
    print("###Distributed Initialized###")
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    num_gpus = torch.cuda.device_count()
    print ("All_GPU", num_gpus)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()

def get_ip(ip_str):
    """
    input format: SH-IDC1-10-5-30-[137,152] or SH-IDC1-10-5-30-[137-142,152] or SH-IDC1-10-5-30-[152, 137-142]
    output format 10.5.30.137
    """
    import re
    # return ".".join(ip_str.replace("[", "").split(',')[0].split("-")[2:])
    return ".".join(re.findall(r'\d+', ip_str)[1:5])
