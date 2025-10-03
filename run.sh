touch bash_history
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.Xauthority \
    -v $HOME/.Xauthority:/tmp/.Xauthority:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ./bash_history:/root/.bash_history \
    --network=host \
    -v .:/GenLoco \
    -v ./torchGenLoco:/torchGenLoco \
    -v ./../genloco-loihi:/genloco-loihi \
    --gpus all \
    lavagenloco \
    bash



