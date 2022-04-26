# Graph Deep Learning Over a Bipartite Graph Using a Biological Pathways Prior

This is the code for the L45 project "Incorporating biological pathways information to aid deep learning on scRNA-set data".

## Run the experiments

To run the experiments in the paper, execute the `benchmarking.py` script.

```
python3 benchmarking.py 
```

Plots can then be created using the `plot_results.py` script with the name of the `csv` file created by the `benchmarking.py` script.

```
python3 plot_results.py experiments/all_models.csv
```

### Run different experiments
There is no command line interface, so to change the experiment setup, you need to edit the `ExperimentConfig` object in the `benchmarking.py` script directly.



