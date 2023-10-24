#!/bin/bash

DATA=$1
OUTDIR=$2
DEVICES=$3
EPOCHS=$4
JOBLEN=$5
NTRIALS=$6
EXPT=$7
STRATEGY="auto"

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

cp $STAGING/$USER/$DATA .
mkdir $HISTORY

HISTORYFILE="$OUTDIR/$EXPT/$EXPT.db"

##### CONDA
cp $STAGING/conda_envs/$TARBALL .
export PATH
mkdir $ENVDIR
tar -xzf $TARBALL -C $ENVDIR
. $ENVDIR/bin/activate
rm $TARBALL

run_pst () {
    pst tune \
        --file $DATA \
	--devices $DEVICES \
	--default-root-dir $OUTDIR \
	--max-epochs $EPOCHS \
	--strategy $STRATEGY \
	--name $EXPT \
	--n-trials $NTRIALS \
        --config config.toml \
	--tuning-dir $HISTORY \
	--max-time $JOBLEN \
	--gradient-clip-algorithm value \
	--gradient-clip-val 1.0 \
	--layer-dropout 0.0 \
	--detect-anomaly
}



clean_up () {
    rm $DATA
    
    if [ -d "$OUTDIR" ]
    then
	tar -czf $OUTDIR.tar.gz $OUTDIR && mv $OUTDIR.tar.gz $STAGING/$USER
        rm -rf $OUTDIR
    fi

    rm -rf $ENVDIR $HISTORY
}

##### run tool and ALWAYS cleanup
run_pst && clean_up || clean_up
