#!/bin/bash

METHODS=("energy" "entropy" "maxlogit" "msp")

BASE_DIR="./outputs"

mkdir -p "${BASE_DIR}/val_results"

for i in "${METHODS[@]}"; do
    echo "========================================="
    echo "Running evaluation on val set for: $i"
    echo "========================================="

    python evaluate_multi_sequence_openset.py \
        --data-root "" \
        --pred-root "${BASE_DIR}/${i}" \
        --yaml "./semantic-kitti.yaml" \
        --sequences 411 421 431 441 451 412 422 432 442 452 \
        --output "${BASE_DIR}/val_results/ood_val_10seq_${i}.json" \
        --closed-prefix "" \
        --anomaly-prefix "${i}_" 

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed on method: $i"
        exit 1
    fi

    echo "Done: $i"
done

echo "All methods completed successfully."
