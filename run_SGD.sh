#!/bin/bash

python main_SGD.py --version 1 --lr 0.1 --num_iters 40000 --decay_factor 2

#python main_SGD.py --version 2 --epsilon 5 --lr 0.1 --num_iters 40000
#
#python main_SGD.py --version 3 --epsilon 5 --lr 0.1 --num_iters 40000
#
#python main_SGD.py --version 4 --epsilon 5 --lr 0.1 --num_iters 40000
#
#python main_SGD.py --version 5 --epsilon 5 --lr 0.1 --num_iters 40000 --momentum 0.9
#
#python main_SGD.py --version 6 --epsilon 5 --lr 0.1 --num_iters 40000 --momentum 0.9 --early_momentum_iters 20000
#
#python main_SGD.py --version 7 --epsilon 5 --lr 0.1 --num_iters 40000 --decay_factor 2
#
#python main_SGD.py --version 8 --epsilon 5 --lr 0.1 --num_iters 40000 --decay_factor 2 --momentum 0.9 --early_momentum_ratio 0.5
