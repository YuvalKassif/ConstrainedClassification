import sys
import torch

# Print the path of the current Python interpreter
print("Python Interpreter Path:", sys.executable)

# Check PyTorch installation and CUDA availability
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("GPU Name:", torch.cuda.get_device_name(0))
