#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate oriMapPaper-env


# Define variables
STARTID=1
N_RUNS=10
SCAN_DIM=128
ENDID=$((STARTID + N_RUNS - 1))

# Base project directory (avoids repetition)
BASE_DIR="/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure_for_paper/figure_04"

EXEC_SCRIPT="${BASE_DIR}/computeTime_new_2026/GPU_predict_orientationMap_synthetic_4DSTEM_via_model.py"
INPUT_FILE_PATH="${BASE_DIR}/sampled_and_stacked/fourD_braggVectors_pointLists/"
OUTPUT_FILE_PATH="${BASE_DIR}/computeTime_new_2026/output_SD${SCAN_DIM}_ID${STARTID}-${ENDID}_GPU/"
RESULT_DIR="${BASE_DIR}/computeTime_new_2026/timing_log_SD${SCAN_DIM}_ID${STARTID}-${ENDID}_GPU"

mkdir -p "$OUTPUT_FILE_PATH" "$RESULT_DIR"

BENCH_LOG="${RESULT_DIR}/summary.txt"

# ... (Metadata logging remains the same for GPU/GPU info) ...

echo -e "\n# Run_ID\tElapsed(s)\tUser_GPU(s)\tSys_GPU(s)" > "$BENCH_LOG"

#-----------------------------------------------------------
# Run N_RUNS serially (one after another) and record timings
#-----------------------------------------------------------

echo "[$(date)] Starting ${N_RUNS} serial runs on $(hostname)..."


for ((i=STARTID; i<STARTID+N_RUNS; i++)); do
    WALL_TIME_FILE="${RESULT_DIR}/timing_SD${SCAN_DIM}_ID${i}.tmp"
    PYTHON_OUTPUT_FILE="${RESULT_DIR}/output_SD${SCAN_DIM}_ID${i}.log"

    echo "[$(date)] Starting run ID $i of $N_RUNS serial runs"

    # Run Python script and capture timing to a temporary file
    /usr/bin/time -f "${i}\t%e\t%U\t%S" \
        -o "$WALL_TIME_FILE" \
        python "$EXEC_SCRIPT" \
            --input_path "$INPUT_FILE_PATH" \
            --output_path "$OUTPUT_FILE_PATH" \
            --run_id "$i" \
            --scan_dim "$SCAN_DIM" > "$PYTHON_OUTPUT_FILE" 2>&1

    # Append ONLY the clean timing line to the summary log
    cat "$WALL_TIME_FILE" >> "$BENCH_LOG"
done

echo ""
echo "[$(date)] py4DSTEM orientation map benchmark completed."
echo ""
echo "Statistics saved in ${BENCH_LOG}"
echo ""
echo "Individual run logs saved in ${RESULT_DIR}/"
echo ""
