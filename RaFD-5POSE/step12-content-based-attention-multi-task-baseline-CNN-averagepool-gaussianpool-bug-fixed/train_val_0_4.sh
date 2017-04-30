#export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

export path=./100-0/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_0.py 2>&1 | tee $fn





export path=./100-1/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_1.py 2>&1 | tee $fn




export path=./100-2/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_2.py 2>&1 | tee $fn




export path=./100-3/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_3.py 2>&1 | tee $fn




export path=./100-4/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_4.py 2>&1 | tee $fn



