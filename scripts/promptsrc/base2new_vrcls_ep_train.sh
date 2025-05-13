#!/bin/bash

# custom config
DATA="/hdd/hdd3/jsh/DATA"
TRAINER=PromptSRC


LOADEP=$1
DATASET=$2
SEED=$3

CFG=vit_b16_c2_ep20_batch4_4+4ctx_vrcls
SHOTS=16
PERCENTAGE=$4

DIR=output/base2new_vrcls_${PERCENTAGE}_ep${LOADEP}/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    /hdd/hdd3/jsh/miniconda3/envs/coop/bin/python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    OPTIM.MAX_EPOCH ${LOADEP} \
    TRAINER.PROMPTSRC.VIRTUAL_CLASS_PERCENTAGE ${PERCENTAGE}
else
    echo "Run this job and save the output to ${DIR}"
    /hdd/hdd3/jsh/miniconda3/envs/coop/bin/python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    OPTIM.MAX_EPOCH ${LOADEP} \
    TRAINER.PROMPTSRC.VIRTUAL_CLASS_PERCENTAGE ${PERCENTAGE}
fi