#! /bin/bash

nohup python -u main.py \
    --gpuId 0 \
    --conv1KWidth 4 \
    --conv1SWidth 20 \
    --num4OutputChannels 2 \
    --num4FirstFC 200 \
    --num4SecondFC 10 \
    --nWeight 25 \
    --path4SaveModel ./files/intermediate/trainedModel0 \
    --path4Summaries ./files/intermediate/summaries \
    --path4SaveEggsFile ./files/intermediate/eggFiles0 \
    1>./files/intermediate/info0.txt 2>&1 &

nohup python -u main.py \
    --gpuId 1 \
    --conv1KWidth 4 \
    --conv1SWidth 20 \
    --num4OutputChannels 2 \
    --num4FirstFC 200 \
    --num4SecondFC 10 \
    --nWeight 26 \
    --path4SaveModel ./files/intermediate/trainedModel1 \
    --path4Summaries ./files/intermediate/summaries \
    --path4SaveEggsFile ./files/intermediate/eggFiles1 \
    1>./files/intermediate/info1.txt 2>&1 &

nohup python -u main.py \
    --gpuId 2 \
    --conv1KWidth 4 \
    --conv1SWidth 20 \
    --num4OutputChannels 2 \
    --num4FirstFC 200 \
    --num4SecondFC 10 \
    --nWeight 27 \
    --path4SaveModel ./files/intermediate/trainedModel2 \
    --path4Summaries ./files/intermediate/summaries \
    --path4SaveEggsFile ./files/intermediate/eggFiles2 \
    1>./files/intermediate/info2.txt 2>&1 &

nohup python -u main.py \
    --gpuId 3 \
    --conv1KWidth 4 \
    --conv1SWidth 20 \
    --num4OutputChannels 2 \
    --num4FirstFC 200 \
    --num4SecondFC 10 \
    --nWeight 28 \
    --path4SaveModel ./files/intermediate/trainedModel3 \
    --path4Summaries ./files/intermediate/summaries \
    --path4SaveEggsFile ./files/intermediate/eggFiles3 \
    1>./files/intermediate/info3.txt 2>&1 &

git commit -a -m "Modify some hyper-parameters."
