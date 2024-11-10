# G&C Network

This project consists in training a feedfoward network using reinforcement learning to fly a drone through a set of gates. For more information, consult the [Nano Drone Racing](https://github.com/fed12345/nano-drone-racing) repository.


## Description
This is a repository to that create, trains and accelerates a nerual network, which handles the drone’s guidance and control functions. This network runs at an impressive rate of 167Hz on an STM32F405 processor, outputting attitude rates and thrust values for an attitude rate PID controller.

The steps that need to be carried out are:
1. Carry out system identification of the drone
2. Train the decided architerue and train NN
3. Evaluate NN
4. Convert it to C code via [ONNX](https://github.com/onnx/onnx)
5. Deploy it on any drone autopilot for example [bitcraze](https://github.com/bitcraze/crazyflie-firmware)(this is not included in this repo)


## Getting Started

### Dependencies

- [Docker](https://docs.docker.com/engine/install/) 
- [Conda](https://www.anaconda.com/download) or Pip

### Installing

1. **Clone the repository:**

   ```bash
   git clone https://github.com/fed12345/controlnet
   cd controlnet
   ```
   

2. **Set Up a Virtual Environment (recommended)**

It’s a best practice to use a virtual environment to manage dependencies. To create a virtual environment, run the following command with conda installed:

```bash
conda create --name controlnet
conda activate controlnet
```

3. **Intall Dependencies**

With the environment active, install all necessary packages by running:

```bash
pip install -r requirements.txt
```

4. **Create Docker Image**

You can build the Docker image by running the following command in the terminal from the directory where the Dockerfile is located:
```bash
docker build -t onnx2c .
```

##  Executing Progam

1. **Model Quadcopter**

It is necessary to aquire some data on the dynamics on the drone, as we will model the droens response as a first over system in terms of roll, pitch, yaw and thrust. We desired states and measured stated. 
We can then use the python modelling script to model the drone, the output will be a series of marticies which have to be inputted in the simulator:

```bash
model_quadcopter.py
```

2. **Training the Model**

The training stage will save model in the results folder as well as some timestamp videos to see how the drone is performing.
```bash
train_godzilla.py
```

3. **Evaluate the model  (Optional)**

Script to run a simulation of the drone controlled by the neural network, this is a quick way to visually evaluate it.
```bash
run_sim.py
```
4. **Convert to ONNX and Quantize Model**

This step will convert the seleced model to ONNX format in order to facilitate quatization and conversion to C code. This step will also quatized the mode for faster inference.
```bash
quantization.py
```

6. **Convert to C code**

This step leverages the docker images created before with the onnx2c library to convert the neural network into optimized c code. 

```bash
docker run --rm -v $(pwd):/module onnx2c /bin/bash -c './onnx2c/build/onnx2c /module/results/quantized/my_model.onnx > /module/results/c_code/output.c'
```




