import onnx
import numpy as np
import matplotlib.pyplot as plt

# Load the ONNX model
model = onnx.load("dummycontrol.onnx")

# Extract the weights
weights = [tensor for tensor in model.graph.initializer]


for tensor in weights:
    # Convert the ONNX tensor to a numpy array
    np_weight = np.frombuffer(tensor.raw_data, dtype=np.float32)#.reshape(tuple(dim.dim_value for dim in tensor.dims))
    # count number of weights under 0.0001
        # count number of weights under 0.0001
    zero_count = np.count_nonzero(np.abs(np_weight) < 0.0001)
    
    print(zero_count)
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(np_weight.flatten(), bins=100, density=False)
    plt.title(f"Weights Distribution for {tensor.name}")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")
plt.show()