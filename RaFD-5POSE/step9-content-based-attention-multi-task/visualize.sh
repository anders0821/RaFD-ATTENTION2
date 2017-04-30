#export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

killall tensorboard
killall chromium-browser
tensorboard --logdir ./ &
sleep 1
chromium-browser http://localhost:6006
#sleep 1000000000
