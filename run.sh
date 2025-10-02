docker run -it --rm \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v .:/GenLoco \
    -v ./torchGenLoco:/torchGenLoco \
    -v ~/genloco-loihi:/genloco-loihi \
    genloco \
    bash \


#    --gpus all \

