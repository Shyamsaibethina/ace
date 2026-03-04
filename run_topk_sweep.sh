#!/bin/bash
set -e

TOP_K_VALUES=(5 10 20)
PIDS=()

for top_k in "${TOP_K_VALUES[@]}"; do
    echo "Launching run with retriever_top_k=${top_k}..."

    python -m eval.finance.run \
        --task_name finer \
        --mode offline \
        --save_path "results/topk_${top_k}" \
        --api_provider openai \
        --generator_model gpt-4o \
        --reflector_model gpt-4o \
        --curator_model gpt-4o \
        --retriever_top_k "${top_k}" \
        > "results/topk_${top_k}.log" 2>&1 &

    PIDS+=($!)
    echo "  PID=$! -> results/topk_${top_k}/ (log: results/topk_${top_k}.log)"
done

echo ""
echo "All ${#TOP_K_VALUES[@]} runs launched in parallel."
echo "Waiting for completion..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    top_k=${TOP_K_VALUES[$i]}
    if wait "$pid"; then
        echo "top_k=${top_k} (PID ${pid}) finished successfully"
    else
        echo "top_k=${top_k} (PID ${pid}) FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
if [ "$FAILED" -eq 0 ]; then
    echo "All runs complete!"
else
    echo "${FAILED} run(s) failed. Check logs."
fi
echo "Results:"
for top_k in "${TOP_K_VALUES[@]}"; do
    echo "  - results/topk_${top_k}/ (log: results/topk_${top_k}.log)"
done
echo "============================================"
