#!/bin/sh
wget -P ./arcface_model https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar
wget https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip
unzip ./checkpoints.zip  -d ./checkpoints
rm checkpoints.zip
wget --no-check-certificate "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip" -O antelope.zip
mkdir -p insightface_func/models
unzip ./antelope.zip -d ./insightface_func/models/
mv ./insightface_func/models/antelopev2 ./insightface_func/models/antelope
rm antelope.zip

mkdir -p ./parsing_model/checkpoint
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O ./parsing_model/checkpoint/79999_iter.pth

wget https://github.com/neuralchen/SimSwap/releases/download/512_beta/512.zip
unzip ./512.zip -d ./checkpoints
rm 512.zip
