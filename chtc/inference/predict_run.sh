#!/bin/bash

DATA=$1
OUTDIR=$2
DEVICES=$3
CHECKPOINT=$4

# simple executable for CHTC
echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
echo 'CUDA_VISIBLE_DEVICES: ' $CUDA_VISIBLE_DEVICES
echo 'Gpu: ' `nvidia-smi -L | grep $CUDA_VISIBLE_DEVICES`

set -e
ENVNAME="pst"
TARBALL="${ENVNAME}.tar.gz"
ENVDIR=$ENVNAME

# Set DDP debug info
export NCCL_DEBUG="INFO"
export TORCH_CPP_LOG_LEVEL="INFO"
export TORCH_DISTRIBUTED_DEBUG="INFO"

##### move data
cp "$STAGING/$USER/$CHECKPOINT" .
cp $STAGING/$USER/$DATA .

LOCAL_CKPT=$(basename $CHECKPOINT)

##### CONDA
cp $STAGING/conda_envs/$TARBALL .
export PATH
mkdir $ENVDIR
tar -xzf $TARBALL -C $ENVDIR
. $ENVDIR/bin/activate
rm $TARBALL

##### run tool
pst predict \
    --file $DATA \
    --devices $DEVICES \
    --checkpoint $LOCAL_CKPT \
    --outdir $OUTDIR 

##### cleanup
rm $DATA $LOCAL_CKPT
mv $OUTDIR $STAGING/$USER
rm -rf $ENVDIR
