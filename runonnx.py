#file to run onnx 

import onnxruntime
import numpy as np

# Load the ONNX model
onnx_model_path = "results/onnx/crazygcnet_0_gates.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# Prepare input data
# Replace with your actual input data
input_data = [[ -2.916933298110962, 2.841855764389038, -0.007863759994506836, -0.17505259811878204, 0.04293742775917053, 0.9415762424468994, 0.08053353428840637, 0.003706212854012847, -0.7468969821929932, 2.09429669380188, 0.9971287250518799, 0.2771109342575073, 0.6462361216545105]]


# Run inference
input_name = session.get_inputs()[0].name
input_data = {input_name: input_data}
output = session.run(None, input_data)

# Print the inference results
print("Output shape:", output[0].shape)
print("Inference results:", output[0])
