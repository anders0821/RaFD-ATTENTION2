#export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

export path=./100_g-0/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_g_0.py 2>&1 | tee $fn





export path=./100_g-1/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_g_1.py 2>&1 | tee $fn




export path=./100_g-2/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_g_2.py 2>&1 | tee $fn




export path=./100_g-3/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_g_3.py 2>&1 | tee $fn




export path=./100_g-4/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

#export CUDA_VISIBLE_DEVICES=""
python -u train_val_g_4.py 2>&1 | tee $fn



