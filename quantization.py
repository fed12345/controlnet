from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantizationMode

# Load the model
onnx_model_path = 'dummycontrol.onnx'
quantized_model_path = 'model_quantized.onnx'

# Quantize
quantized_model = quantize_dynamic(onnx_model_path, quantized_model_path)
