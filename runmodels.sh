#!/bin/bash

python resWgisd.py -t test -m fpn -l 0 -D wgisd
python resWgisd.py -t test -m fpn -l 1 -D wgisd
python resWgisd.py -t test -m fpn -l 5 -D wgisd
python resWgisd.py -t test -m untrained -l 5 -D wgisd
python resWgisd.py -t test -m byol -l 0 -D wgisd
python resWgisd.py -t test -m byol -l 1 -D wgisd
python resWgisd.py -t test -m byol -l 5 -D wgisd
python resWgisd.py -t test -m resnet -l 0 -D wgisd
python resWgisd.py -t test -m resnet -l 1 -D wgisd
python resWgisd.py -t test -m resnet -l 5 -D wgisd
