#!/bin/bash
#


gradient deployments create \
    --deploymentType TFServing \
    --modelId moswd2hbuslslrd \
    --name "LSTM128-dr0-rdr-2-batch32-lr001" \
    --machineType G1 \
    --imageUrl tensorflow/serving \
    --instanceCount 1

