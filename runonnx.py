#file to run onnx 

import onnxruntime
import numpy as np

# Load the ONNX model
onnx_model_path = "/home/federico/crazyFly/controlnet/results/onnx/crazygcnet_0_gates.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# Prepare input data
# Replace with your actual input data
input_data = [ -2.8718018531799316, 2.8313040733337402, -0.07401180267333984, -0.17595459520816803, 0.044475115835666656, 0.023019619286060333, -0.041953444480895996, -0.05448269471526146, -0.7612579464912415, -0.22741959989070892, -0.18580913543701172, -0.004937510937452316, 7.31462287902832]


# Run inference
input_name = session.get_inputs()[0].name
input_data = {input_name: input_data}
output = session.run(None, input_data)

# Print the inference results
print("Output shape:", output[0].shape)
print("Inference results:", output[0])
