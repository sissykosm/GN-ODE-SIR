# Neural Ordinary Differential Equations for Modeling Epidemic Spreading (GN-ODE)
## Predicting SIR model on Large Complex Networks

Network Datasets are in the ```real_graphs``` folder.

SIR labels for $t \in [0,...T]$ are extracted using Monte-Carlo simulations on the undirected/unweighted preprocessed graphs.

![This is an image](./images/sir_predictions_karate.png)

<sub>Visualization of the evolution of infection over time on the karate dataset. Given the initially infected nodes (with red at t = 0), we compare the predictions (probability that a node is in state I) of the proposed GN-ODE model (right) against the ground truth probabilities obtained through Monte-Carlo simulations (left).</sub>

