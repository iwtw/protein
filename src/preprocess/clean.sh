#!/bin/bash
python ./get_single_cell_crop_csv.py ../../data/train.csv ../../data/train_single_cell_crop ../../data/train_single_cell_crop.csv
python ./get_single_cell_crop_csv.py ../../data/test.csv ../../data/test_single_cell_crop ../../data/test_single_cell_crop.csv
