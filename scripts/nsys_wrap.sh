#!/bin/bash

if [[ -z "${NSYS_OUTPUT}" ]]; then
    NSYS_OUTPUT=prof
fi
if [[ -z "${NSYS_NODE_INTERVAL}" ]]; then
    NSYS_NODE_INTERVAL=1
fi
if [ "${SLURM_LOCALID}" -eq 0 ] && [ "$(( SLURM_NODEID % NSYS_NODE_INTERVAL ))" -eq 0 ];
then
    nsys profile \
        -f true \
        -o ${NSYS_OUTPUT}_node${SLURM_NODEID} \
        -c cudaProfilerApi \
        --trace cuda,nvtx,cudnn,osrt \
        --export sqlite \
        $@
else
    $@
fi
sleep 30  # wait for nsys to complete
