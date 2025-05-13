#!/bin/bash


# custom config
DATA="/hdd/hdd3/jsh/DATA"
TRAINER=PromptSRC

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep50_batch4_4+4ctx_vrcls
SHOTS=16
LOADEP=50
PERCENTAGE=$3
SUB=$4

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new_vrcls_${PERCENTAGE}_ep50/train_base/${COMMON_DIR}
DIR=output/bbase2new_vrcls_${PERCENTAGE}_ep50/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    /hdd/hdd3/jsh/miniconda3/envs/coop/bin/python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.PROMPTSRC.VIRTUAL_CLASS_PERCENTAGE 0

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    /hdd/hdd3/jsh/miniconda3/envs/coop/bin/python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.PROMPTSRC.VIRTUAL_CLASS_PERCENTAGE 0
fi