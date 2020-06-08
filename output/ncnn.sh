#!/bin/bash
python -m onnxsim $1.onnx $1-sim.onnx
cp $1-sim.onnx /home/night/PycharmProjects/ncnn/ncnn/build/tools/onnx/$1-sim.onnx
cd /home/night/PycharmProjects/ncnn/ncnn/build/tools/onnx/
./onnx2ncnn $1-sim.onnx $1.param $1.bin
cp $1.param /home/night/PycharmProjects/ncnn/ncnn/build/tools/$1.param
cp $1.bin /home/night/PycharmProjects/ncnn/ncnn/build/tools/$1.bin
cd /home/night/PycharmProjects/ncnn/ncnn/build/tools
./ncnnoptimize $1.param $1.bin $1-opt.param $1-opt.bin 0