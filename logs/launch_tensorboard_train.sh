# Check most recent dir
logdir=$(ls -td ./train/* | head -n 1)
# Check env var
SERVER_MODE=${TENSORBOARD_SERVER:-0}
if [ $SERVER_MODE -eq 1 ]; then
    # fixed port
    port=6123
    echo "Starting tensorboard: $logdir"
    tensorboard --logdir=$logdir --port=$port --bind_all
else
    # random port
    port=$(((RANDOM%1000)+6006))
    echo "Starting tensorboard: $logdir"
    open http://localhost:$port
    tensorboard --logdir=$logdir --port=$port
fi
