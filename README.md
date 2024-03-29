# Neural Ordinary Differential Equations for Modeling Epidemic Spreading (GN-ODE)
## Predicting SIR model on Large Complex Networks

![This is an image](./images/sir_predictions_karate.png)

<sub>Visualization of the evolution of infection over time on the karate dataset. Given the initially infected nodes (with red at t = 0), we compare the predictions (probability that a node is in state I) of the proposed GN-ODE model (right) against the ground truth probabilities obtained through Monte-Carlo simulations (left).</sub>

Network Datasets are in the ```real_graphs``` folder.

SIR labels for $t \in [0,...T]$ are extracted using Monte-Carlo simulations on the undirected/unweighted preprocessed graphs.

For **single graph training**: `python monitorer-sim.py`  
Inside the script you need to specify the following:
- `datasets_array` can be equal to `['./real_graphs/karate']` or based on the desired dataset path in folder `real_graphs`.
- `model` can be equal to `'dmp'` (Dynamic Message Passing), `'GCN'`, `'GIN'` (GNN variants), `'ode_nn'` (the proposed GN-ODE).
- the hyperparameters for training: `epochs`, `lr`, `batch_size`, and the `hidden_dim_array` (e.g. [64] for hidden_size 64).
- the hyperparameters for Monte-carlo simulations and labels extraction: `deltaT` (step size of the ODE solver), `maxTime` (number of spreading time steps), `sim` (number of Monte-Carlo simulations). The parameters `beta`, `gamma` (infection and recovery rates) in the current script are sampled randomly from uniform distribution.
- `out_of_dist` equal to `True` for within distribution values of parameters `beta`, `gamma` or equal to `False` for out of distribution experiments. Also, `trials_per_number` is the number of samples to extract from each graph. For the number of initial set of infected nodes that are randomly sampled set a value in the list `n_I` (e.g. [2] for 2 initial infected nodes).

The `ode_nn` script contains functions for **label extraction based on Monte-Carlo simulations** using networkx `def sir_nx`, pandas `def sir_pandas` and torch vectors `def sir_torch` (that runs faster using gpu).
- In the folder `multi-graph-1` we give some example labels for the karate network, with random initial set of infected nodes equal to 2 and `beta`, `gamma` sampled randomly as shown in script `python monitorer-sim.py` and 10000 simulations.
- By running `python monitorer-sim.py` without changes on the data parameters, those labels are used by default by the specified model, e.g. the proposed if `model = 'ode_nn'`.
- For the rest of datasets, or different initial set of infected nodes and Monte-Carlo parameters, you automatically follow the label extraction process by running `python monitorer-sim.py` and then the specified model runs on the extracted data.

For **multiple graph training and inference on a bigger unseen graph**: `python monitorer-ngraphs.py`  
Inside the script you need to specify the following:
- `datasets_array` can be equal to `['./real_graphs/dolphins+fb-food+fb-social+openflights+wiki-vote+epinions']` for training on dolphins, fb-food, fb-social, openflights, wiki-vote and predictions on epinions.
- rest parameters can be selected as descrived above for `python monitorer-sim.py`.