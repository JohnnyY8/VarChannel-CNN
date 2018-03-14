#! /bin/bash

nohup python -u main.py \
    --gpuId 0 \
    --conv1KWidth 5 \
    --conv1SWidth 2 \
    --num4OutputChannels 12 \
    --path4SaveModel ./files/intermediate/trainedModel0 \
    --path4Summaries ./files/intermediate/summaries0 \
    --path4SaveEggsFile ./files/intermediate/eggFiles0 \
    1>./files/intermediate/info0.txt 2>&1 &

nohup python -u main.py \
    --gpuId 1 \
    --conv1KWidth 3 \
    --conv1SWidth 3 \
    --num4OutputChannels 8 \
    --path4SaveModel ./files/intermediate/trainedModel1 \
    --path4Summaries ./files/intermediate/summaries1 \
    --path4SaveEggsFile ./files/intermediate/eggFiles1 \
    1>./files/intermediate/info1.txt 2>&1 &
