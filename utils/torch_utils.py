import torch

def print_visible_cuda_devs():
    # print the list of visible cuda devices
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available.")

if __name__ == '__main__':
    print_visible_cuda_devs()