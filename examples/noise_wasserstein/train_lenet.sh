#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/noise_wasserstein/lenet_solver_wass.prototxt
