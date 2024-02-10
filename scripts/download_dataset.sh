#!/bin/bash

# Create required storage folders
mkdir -p storage/football_data/;

# Download data
curl -L https://www.dropbox.com/sh/vc7mbdhnt4bt8vc/AACcvA_7Ly-QcRE4lv40o3zAa?dl=1 -o ../storage/data.zip;
unzip ../storage/data.zip -d storage/football_data/;
rm ../storage/data.zip;
rm -r ../storage/football_data/splits/;

unzip ../storage/football_data/video_db.zip -d ../storage/football_data;
unzip ../storage/football_data/finegrained_txt_db.zip -d ../storage/football_data;
rm ../storage/football_data/*.zip;