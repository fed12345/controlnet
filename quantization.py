from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantizationMode
from stable_baselines3 import PPO
import torch
import torch.nn as nn
# first test only moment randomization
path = 'models/776000000_fast_yaw.zip'
onnx_model_path = 'results/onnx/yaw.onnx'
quantized_model_path = 'results/quantized/yaw.onnx'
model = PPO.load(path)

# get network
network = list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]
network = nn.Sequential(*network)
dummy_input = torch.randn(1, 13)
pytorch_total_params = sum(p.numel() for p in network.parameters())

torch.onnx.export(
    network,
    dummy_input, 
    onnx_model_path,
    do_constant_folding=True, 
    input_names=["input1"], 
    output_names=["output1"], 
    export_params=True, 
    verbose=False
    )

print('NETWORK:')
print(pytorch_total_params)

print(model.policy.action_dist)
print(model.policy.log_std)
print(model.policy.log_std.exp())
network_std = model.policy.log_std.exp().cpu().detach().numpy()
print(network_std)

# Quantize
quantized_model = quantize_dynamic(onnx_model_path, quantized_model_path)
