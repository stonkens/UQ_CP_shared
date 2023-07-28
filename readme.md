

## Pipeline


## Models

Because models might take in very different data, we write a data pipeline for each model (if needed). 

Each model should also have a predict function that outputs results in (ground truth, prediction) tuples into a output file for downstream evaluation and visualization.

The output results are stored in the ```output``` folder in the format of ```<dataset>-<model>_out.pkl```.

## UQ

A bunch of UQ algorithms. For post-hoc methods, it takes in a output pickle file. 