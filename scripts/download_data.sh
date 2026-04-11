#!/usr/bin/env bash
set -e

mkdir -p dataset/librispeech dataset/esc50

echo "Downloading LibriSpeech train-clean-100..."
wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz -P dataset/
tar -xzf dataset/train-clean-100.tar.gz -C dataset/librispeech/ --strip-components=1
rm dataset/train-clean-100.tar.gz

echo "Downloading ESC-50..."
wget -c https://github.com/karoldvl/ESC-50/archive/master.zip -P dataset/
unzip -q dataset/master.zip -d dataset/
mv dataset/ESC-50-master/* dataset/esc50/
rmdir dataset/ESC-50-master
rm dataset/master.zip

echo "Done. Data is in dataset/librispeech/ and dataset/esc50/"
