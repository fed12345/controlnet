# Use the official Ubuntu 22.10 image as a base
FROM ubuntu:22.04

RUN  apt-get update -q && apt-get upgrade -y -q --no-install-recommends && apt-get clean
RUN apt-get install -y libprotobuf-dev protobuf-compiler=3.12.*
RUN apt-get install -y git cmake 
RUN git clone https://github.com/kraiskil/onnx2c.git
RUN cd onnx2c
RUN git submodule update --init
RUN mkdir build
RUN cd build
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make onnx2c
CMD ["/bin/bash"]
