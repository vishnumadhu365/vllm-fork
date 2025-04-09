#!/bin/bash

__fp8="no"

while [ -n "$1" ];
    do
        case $1 in
        -fp8 )
            __fp8="yes"
            ;;
        esac
        shift
    done

model=/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B
model_short=$(basename $model)
#replace all '.' with '-'
model_short=${model_short//./-}
tmp_file_name=$(mktemp /tmp/_benchmark_${model_short}_XXXXXXX)
error_log_file="${tmp_file_name}_error.log"
log_file="${tmp_file_name}.log"

# Generate an empty result file.
# This way in case of any crash it will be treated by jenkins as failure
if [[ -n "$TEST_RESULTS_DIR" ]]; then
    mkdir -p ${TEST_RESULTS_DIR}
    LOG_PATH=$(mktemp ${TEST_RESULTS_DIR}/benchmark_${model_short}_XXXXXX.xml)
fi

scenario=fp8
if [[ $__fp8 == "no" ]]; then
    scenario=bf16
fi

# Get threshold values according to the scenario and env variables
throughput_threshold=999999

IFS=',' read -ra pairs <<< "$PERF_THRESHOLD"
for pair in "${pairs[@]}"; do
    IFS='=' read -ra kv <<< "$pair"
    key="${kv[0]}"
    value="${kv[1]}"

    if [[ $key == $scenario ]]; then
        throughput_threshold=$value
    fi
done

warmup_threshold=1

IFS=',' read -ra pairs <<< "$WARMUP_THRESHOLD"
for pair in "${pairs[@]}"; do
    IFS='=' read -ra kv <<< "$pair"
    key="${kv[0]}"
    value="${kv[1]}"

    if [[ $key == $scenario ]]; then
        warmup_threshold=$value
    fi
done

# Get the directory of the current script
script_dir=$(dirname "$(readlink -f "$0")")

start=`date +%s`

if [[ $__fp8 == "yes" ]]; then
    export QUANT_CONFIG=/software/users/kpietkun/configs/llama3.1_quant_cofnig.json
    python  $script_dir/../../benchmarks/benchmark_throughput.py \
        --model $model \
        --device hpu \
        --seed 2024 \
        --backend vllm \
        --dataset /mnt/weka/data/pytorch/llama2/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000 \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 128 \
        --quantization=inc \
        --kv-cache-dtype=fp8_inc \
        --weights-load_device=cpu \
        --use-padding-aware-scheduling 2> >(tee -a $error_log_file) | tee -a $log_file

else

    python  $script_dir/../../benchmarks/benchmark_throughput.py \
        --model $model \
        --device hpu \
        --seed 2024 \
        --backend vllm \
        --dataset /mnt/weka/data/pytorch/llama2/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000 \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 128 \
        --use-padding-aware-scheduling 2> >(tee -a $error_log_file) | tee -a $log_file

fi

end=`date +%s`
runtime=$((end-start))
printf " -------------- \nBenchmark took: %2d:%02d\n\n" $((runtime/60)) $((runtime%60)) 

throughput=$(grep -oP 'Throughput: [0-9.]+ requests/s, \K[0-9]+' $log_file)
warmup=$(grep -oP 'Warmup finished in +\K[0-9:]+' $log_file)

warmup_status="FAILED"
warmup_fail=1
if [[ "$warmup" ]]; then
    if ((warmup <= warmup_threshold)); then
        warmup_status="PASSED"
        warmup_fail=0
    fi
fi
echo "=== $warmup_status warmup MODEL: ${model_short}  ($warmup <= $warmup_threshold) ==="
throughput_status="FAILED"
throughput_fail=1
if [[ "$throughput" ]]; then
    if ((throughput >= throughput_threshold)); then
        throughput_status="PASSED"
        throughput_fail=0
    fi
fi
echo "=== $throughput_status throughput MODEL: ${model_short}  ($throughput >= $throughput_threshold) ==="

if [[ -s "$error_log_file" ]]; then
    runtime_error=1
else
    runtime_error=0
fi

if [[ -n "$TEST_RESULTS_DIR" ]]; then
    # Store full benchmark log
    chmod +r $log_file
    mv $log_file ${TEST_RESULTS_DIR}/

    # Report results for jenkins
    cat <<EOF > ${LOG_PATH}

<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="benchmark" errors="$runtime_error" failures="$((throughput_fail + warmup_fail))" skipped="0" tests="3" time="$runtime">
<testcase classname=".jenkins.benchmark.${model_short}-bf16" name="${model_short}-bf16-no-runtime-error" time="$runtime">
EOF
    if [[ "$RUNTIME_ERROR" -eq 1 ]]; then
        cat <<EOF >> ${LOG_PATH}
<failure message="Runtime error"> $(cat $error_log_file) </failure>
EOF
    fi
 cat <<EOF >> ${LOG_PATH}
</testcase>
<testcase classname=".jenkins.benchmark.${model_short}-bf16" name="${model_short}-bf16-throughput" time="$runtime">
<properties>
<property name="throughput" value="$throughput"/>
<property name="throughput threshold" value="$throughput_threshold"/>
</properties>
EOF
    if [[ "$throughput_fail" -eq 1 ]]; then
        cat <<EOF >> ${LOG_PATH}
<failure message="Throughput did not meet the threshold  ($throughput &lt; $throughput_threshold)"></failure>
EOF
    fi
 cat <<EOF >> ${LOG_PATH}
</testcase>
<testcase classname=".jenkins.benchmark.${model_short}-bf16" name="${model_short}-bf16-warmup" time="$warmup">
<properties>
<property name="warmup time" value="$warmup"/>
<property name="warmup threshold" value="$warmup_threshold"/>
</properties>
EOF
    if [[ "$warmup_fail" -eq 1 ]]; then
        cat <<EOF >> ${LOG_PATH}
<failure message="Warmup did not meet the threshold ($warmup &gt; $warmup_threshold)"></failure>
EOF
    fi
    cat <<EOF >> ${LOG_PATH}
</testcase>
</testsuite>
</testsuites>
EOF

fi
