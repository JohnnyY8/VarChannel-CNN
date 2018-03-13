#! /bin/bash

nohup python -u main.py \
    --gpuId 0 \
    --conv1KWidth 20 \
    --conv1SWidth 5 \
    --num4OutputChannels 20 \
    --path4SaveModel ./files/intermediate/trainedModel0 \
    --path4Summaries ./files/intermediate/summaries0 \
    --path4SaveEggFiles ./files/intermediate/eggFiles0 \
    1>./files/intermediate/info0.txt 2>&1 &

nohup python -u main.py \
    --gpuId 1 \
    --conv1KWidth 30 \
    --conv1SWidth 10 \
    --num4OutputChannels 30 \
    --path4SaveModel ./files/intermediate/trainedModel1 \
    --path4Summaries ./files/intermediate/summaries1 \
    --path4SaveEggFiles ./files/intermediate/eggFiles1 \
    1>./files/intermediate/info1.txt 2>&1 &
