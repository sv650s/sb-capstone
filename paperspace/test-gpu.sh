#!/bin/bash
# Training Reference: https://docs.paperspace.com/gradient/tutorials/train-a-model-with-the-cli
# Machine Types: https://support.paperspace.com/hc/en-us/articles/234711428-Machine-Pricing
# Free instance: https://docs.paperspace.com/gradient/instances/free-instances?_ga=2.254671808.999355169.1587737794-211442023.1587536380
#       C3 (CPU) or P4000 (GPU)
#    --container vtluk/paperspace-tf-gpu:1.0 \


experiment_name=test-gpu
machine_type=P4000
tf_version=`grep FROM.*tensorflow Dockerfile  | awk -F: '{print $2}'`

echo "Running experiment with following command"
echo "python train/test-gpu.py"
echo "machine_type: ${machine_type}"
echo "experiment_name: ${experiment_name}"
echo "tf_version: ${tf_version}"

gradient experiments run singlenode \
    --name ${experiment_name} \
    --projectId pr1cl53bg \
    --machineType ${machine_type} \
    --container vtluk/paperspace-experiment:${tf_version} \
    --command "python train/test-gpu.py" \
    --workspace . \


echo "Finished running experiment with following command"
echo "python train/test-gpu.py"
echo "machine_type: ${machine_type}"
echo "experiment_name: ${experiment_name}"
echo "tf_version: ${tf_version}"


