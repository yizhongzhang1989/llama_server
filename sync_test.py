import os
import signal
import torch  
import torch.distributed as dist  


def signal_handler(signum, frame):  
    pass

def get_rank_pid():
    # return a list of pid of all processes of all rands    

    # create a list of world_size elements
    world_size = dist.get_world_size()
    pid_list = [0] * world_size

    # get the pid of the current process
    rank = dist.get_rank()
    pid = os.getpid()
    pid_list[rank] = pid

    print(f"Rank {rank}, world_size {world_size}, pid {pid}")
        
    # broadcast the pid of the current process to all other processes
    for i in range(world_size):        
        t_pid = torch.tensor([pid], dtype=torch.long).cuda(rank)
        dist.broadcast(t_pid, src=i)
        pid_list[i] = t_pid.item()

    return pid_list


def main():  
    dist.init_process_group(backend='nccl')  
  
    rank = dist.get_rank()  
    torch.cuda.set_device(rank)  

    pid_list = get_rank_pid()

    print(f"Rank {rank} has pid_list {pid_list}")

    while True:

        print(f"Rank {rank} is waiting for input")

        # get input from user in rank 0
        if rank == 0:
            userText = input('Enter a string: ')            
            # For safety, you might want to ensure the text is not too long,  
            # or handle that case appropriately before broadcasting.  

            os.kill(pid_list[1], signal.SIGUSR1)
        else:              
            userText = None

            signal.signal(signal.SIGUSR1, signal_handler)
            signal.pause()


        print(f"Rank {rank} received input: {userText}")

        # Assuming userText is a string, we need to convert it to a tensor for broadcasting.
        # One way is to encode it to bytes, get the length, and then create two tensors:
        # one for the length and one for the byte values.
        if rank == 0:
            userText_bytes = userText.encode()
            length = torch.tensor([len(userText_bytes)], dtype=torch.long).cuda()
            data = torch.tensor(list(userText_bytes), dtype=torch.uint8).cuda()
        else:
            length = torch.tensor([0], dtype=torch.long).cuda()
            # Allocate tensor based on maximum expected length, or use dynamic resizing for efficiency.
            data = torch.empty(1024, dtype=torch.uint8).cuda()

        print(f"Rank {rank} broadcasting")

        # Broadcast the length first, then the actual data.
        dist.broadcast(length, src=0)
        dist.broadcast(data[:length.item()], src=0)

        print(f"Rank {rank} broadcasted")

        if rank != 0:
            userText = bytes(data[:length.item()]).decode()

        print(f"Rank {rank} received string: {userText}")

        # wait for all ranks to finish
        dist.barrier()

        print(f"Rank {rank} is done")

        if userText == "exit":
            break


if __name__ == "__main__":  
    main()  
