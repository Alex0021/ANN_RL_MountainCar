# random port
port=$(((RANDOM%1000)+6006))
logdir=$(ls -td ./eval/* | head -n 1)
# logdir="./logs/*"
echo "Starting tensorboard: $logdir"
open http://localhost:$port
tensorboard --logdir=$logdir --port=$port