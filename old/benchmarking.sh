#!/bin/bash

python3 benchmark.py mnist_keras_conv_test.py
python3 benchmark.py mnist_keras_noise_shuffle.py
python3 benchmark.py mnist_keras_noise_test.py

for i in {1..20}
do
	python3 benchmark.py mnist_keras_size_test.py 10
	python3 benchmark.py mnist_keras_epoch_test.py 20
	python3 benchmark.py mnist_keras_dataset_test.py 10
done
