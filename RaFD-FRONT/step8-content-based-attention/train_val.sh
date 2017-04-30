#export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

export path=./100/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val.py 2>&1 | tee $fn

