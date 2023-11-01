from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantizationMode
from stable_baselines3 import PPO
import torch
import torch.nn as nn
# first test only moment randomization
path = 'models/Benchmark/315000000_175.zip'
model = PPO.load(path)

# get network
network = list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]
network = nn.Sequential(*network)
dummy_input = torch.randn(1, 13)
pytorch_total_params = sum(p.numel() for p in network.parameters())

torch.onnx.export(
    network, # pytorch model (with the weights)
    dummy_input, # model input (or a tuple for multiple inputs)
    "results/onnx/175.onnx", # where to save the model
    do_constant_folding=True, # whether to execute constant folding for optimization
    input_names=["input1"], # the model's input names
    output_names=["output1"], # the model's output names
    export_params=True, # store the trained parameter weights
    verbose=False
    )

print('NETWORK:')
print(pytorch_total_params)

print(model.policy.action_dist)
print(model.policy.log_std)
print(model.policy.log_std.exp())
network_std = model.policy.log_std.exp().cpu().detach().numpy()
print(network_std)

# Load the model
onnx_model_path = 'results/onnx/175.onnx'
quantized_model_path = 'results/quantized/175.onnx'

# Quantize
quantized_model = quantize_dynamic(onnx_model_path, quantized_model_path)
