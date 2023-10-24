#!/bin/bash

DATA=$1
OUTDIR=$2
DEVICES=$3
EPOCHS=$4
JOBLEN=$5

### these were the results of the current best run
# simple executable for CHTC
echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
echo 'CUDA_VISIBLE_DEVICES: ' $CUDA_VISIBLE_DEVICES
echo 'Gpu: ' `nvidia-smi -L | grep $CUDA_VISIBLE_DEVICES`
echo 'Command: ' ${@}

set -e
ENVNAME="pst"
TARBALL="${ENVNAME}.tar.gz"
ENVDIR=$ENVNAME

# Set DDP debug info
export NCCL_DEBUG="INFO"
export TORCH_CPP_LOG_LEVEL="INFO"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export CUDA_LAUNCH_BLOCKING="1"

##### move data
cp $STAGING/$USER/$DATA .
HISTORYDIR="tuning_history"

case $DATA in
    esm2_t6_8M*)
	HISTORY="esm2_t6_8M"
	;;
    esm2_t12_35M*)
	HISTORY="esm2_t12_35M"
	;;
    esm2_t30_150M*)
	HISTORY="esm2_t30_150M"
	;;
    esm2_t33_650M*)
	HISTORY="esm2_t33_650M"
	;;
    *)
	HISTORY="default"
	;;
esac
 
HISTORYFILE=$HISTORYDIR/$HISTORY/"history.db"
cp $STAGING/$USER/$HISTORYFILE .

##### CONDA
cp $STAGING/conda_envs/$TARBALL .
export PATH
mkdir $ENVDIR
tar -xzf $TARBALL -C $ENVDIR
. $ENVDIR/bin/activate
rm $TARBALL

run_pst () {
    pst train \
        --file $DATA \
	--devices $DEVICES \
	--default-root-dir $OUTDIR \
	--max-epochs $EPOCHS \
	--train-on-full \
	--from-study history.db \
	--max-time $JOBLEN \
	--use-scheduler \
	--config config.toml \
	--gradient-clip-val 1.0 \
	--gradient-clip-algorithm value \
	--strategy "auto" \
	--warmup-steps 3600 
}

##### run tool
run_pst

##### cleanup
rm $DATA history.db
tar -czf $OUTDIR.tar.gz $OUTDIR
mv $OUTDIR.tar.gz $STAGING/$USER
rm -rf $ENVDIR $OUTDIR
