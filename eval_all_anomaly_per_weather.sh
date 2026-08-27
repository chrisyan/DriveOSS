#!/bin/bash

METHODS=("energy" "entropy" "maxlogit" "msp" "rba")

declare -A WEATHER_SCENES
WEATHER_SCENES["Day"]="411 412"
WEATHER_SCENES["Night"]="421 422"
WEATHER_SCENES["Rain"]="441 442"
WEATHER_SCENES["Cloudy"]="431 432"
WEATHER_SCENES["Snow"]="451 452"

BASE_DIR=""

mkdir -p "${BASE_DIR}/val_results"

for weather in "${!WEATHER_SCENES[@]}"; do
    sequences=${WEATHER_SCENES[$weather]}
    weather_lower=$(echo "$weather" | tr '[:upper:]' '[:lower:]')

    for i in "${METHODS[@]}"; do
        echo "========================================="
        echo "Running evaluation on val set: weather=${weather}, method=${i}"
        echo "========================================="

        python evaluate_multi_sequence_openset.py \
            --data-root "" \
            --pred-root "${BASE_DIR}/${i}" \
            --yaml "./semantic-kitti.yaml" \
            --sequences ${sequences} \
            --output "${BASE_DIR}/val_results/ood_val_${weather_lower}_${i}.json" \
            --closed-prefix "" \
            --anomaly-prefix "${i}_" 


        if [ $? -ne 0 ]; then
            echo "ERROR: Failed on weather=${weather}, method=${i}"
            exit 1
        fi

        echo "Done: weather=${weather}, method=${i}"
    done
done

echo "All combinations completed successfully."