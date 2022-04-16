# L45 Coursework

## TODO
### Edge prediction
- [ ] Write training loop to train MPNN on edge prediction with masked values on the train set
- [ ] Make MPNN work for edge prediction
### Node Predicition
- [X] Make MPNN work for node prediction
- [X] Train MPNN to work on node prediction with some masked values
### Benchmarking
- [ ] Measure accuracy of node prediction task
- [ ] Measure accuracy of edge prediction task
- [ ] Create plots that look at specific values the model predicts to check for errors
- [ ] Find a nice graph to show results
- [ ] Find a suitable benchmark to test it against (Autoencoder? PCA?)
- [ ] Figure out what the applications of this is and steer benchmarking accordingly
### Data
- [o] Find dataset with labels to use for cell classification
    - [X] Assign unique ID to each gene so model knows which genes are which
    - [ ] Add information to the gene features from different dataset
- [ ] Integrate multiple datasets
