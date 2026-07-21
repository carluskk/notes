Paper: _SAGE: Sustainable and green energy-aware client selection for federated learning in IoT_, M. Savoia et al., 2026

> [!IMPORTANT] Summary
> SAGE is a federated-learning client-selection strategy that, instead of randomly choosing clients, selects devices by jointly considering their remaining battery, the diversity of their local data, and how much renewable energy they are currently using. It solves an optimization problem each round to choose the best eligible clients, while protecting low-energy devices by excluding those below 20% battery and reducing local training for those below 30%. The core goal is to make federated learning more energy-efficient and sustainable without sacrificing the data diversity needed to train an accurate global model.

## Overview

SAGE (Sustainable And Green Energy-aware client selection for Federated Learning in IoT) is an energy-aware CS strategy for FL in distributed IoT environments. The method considers:
- The residual energy available for each client.
- The Jensen–Shannon divergence as a statistical measure of data dissimilarity.
- The proportion of renewable energy contributing to the client’s power supply. 
By incorporating these three aspects into the selection process, the proposed approach ensures a balanced participation of clients, reduces the risk of premature energy depletion, and prefers the use of renewable energy in distributed training.

## Methods

- **Residual Energy**: Clients are assessed according to their remaining power levels. The optimization process prioritizes devices with higher energy availability, while those with critically low reserves, if selected, are handled in a way that reduces their consumption by limiting the number of local epochs.
- **Jensen-Shannon Divergence (JSD)**: Symmetric measure that ==quantifies the similarity between probability distributions==. For each client they compute the average divergence with all others, obtaining a measure of how much unique information it contributes. Clients with higher JSD values are prioritized. The JSD between two different clients $P_i$ and $P_j$ is defined as: $$JSD(P_{i}||P_{j}) = \frac{1}{2} KL(P_{i}||M) + \frac{1}{2} KL(P_{j}||M)$$
  where $M=\frac{1}{2}(P_{i}+ P_{j})$ is the mean distribution and $KL(\cdot||\cdot)$ is the Kullback-Leibler divergence.
- **Renewable energy**: Clients are evaluated based on the proportion of energy sourced from renewable resources.

To avoid numerical issues when a client has no samples from a particular class, **additive (Laplace) smoothing** is applied. A small value $\epsilon$ is added to the count of every class before computing the probability distribution.

For example, class counts $[5, 5, 0]$ become $[5+\epsilon, 5+\epsilon, \epsilon]$. This avoids zero probabilities in the divergence calculation.

Let $y_{i}\in{0, 1}$ be a binary decision variable indicating whether client $i$ is selected. If $N$ is the total number of clients, $R_{\max}$ is the total number of communication rounds, and $r$ is the current round, the CS strategy is formulated as the following optimization problem:

$$
\max \sum_{i=1}^{N} y_i \left(aE_i+bD_i+cR_i\right)
$$
- Select at most $k+r$ clients. The maximum number of participating clients increases with the communication round $r$. _”This evolution is strategically designed for Non-IID scenarios: ==early rounds prioritize the most robust and green clients to stabilize the model quickly==. In later rounds, the expanded capacity allows the inclusion of other clients, who may have rare data distributions but poorer energy profiles, thereby refining the global model’s accuracy and generalization without wasting resources at the start”_.
- Select at most $20\%$ of the total clients from the subset $C'$.
- Select at least $k$ clients in each communication round.
- The round index $r$ must satisfy $0 \leq r < R_{\max}$.
- Clients must have at least $20\%$ residual energy ($E_i \geq 0.2$).
- Each client is either selected ($y_i=1$) or not selected ($y_i=0$).
- The weights satisfy $a+b+c=1$.

The choice of weights should align with specific application goals:
- Energy-Prioritized Mode: assigning higher values to $a$ and $c$ prioritizes network longevity and sustainability over convergence speed.
- Accuracy-Prioritized Mode: increasing $b$ ensures that clients with diverse datasets are selected, improving the generalization of the global model.

### Algorithm

- Initialize the global model $w^0$, the communication round $r=0$, and the set of $N$ clients.
- For each client $i$, assign:
	- residual energy $E_i$,
	- data divergence $D_i$,
	- renewable energy ratio $R_i$.
- Set the optimization weights $a$, $b$, and $c$.

<u>At Each Communication Round</u>

1. **Update client information**
	- Update each client's residual energy $E_i$ based on previous energy consumption.
	- Update the renewable energy ratio $R_i$ to simulate changing renewable-energy availability.
	- Clients with $E_i < 0.2$ are excluded from the candidate set.
2. **Select clients**
	- Solve the SAGE optimization problem using $E_i$, $D_i$, and $R_i$.
	- Select a set of clients $S_{\mathrm{sel}}$ satisfying: $$k \leq |S_{\mathrm{sel}}| \leq \min(k+r,\;0.2N)$$
3. **Perform local training**
	- Each selected client normally trains for $L$ local epochs.
	- If a selected client has low residual energy ($E_i < 0.3$), its number of local epochs is reduced: $$L_i=\max\left(1,\left\lfloor\frac{L}{2}\right\rfloor\right)$$
	- After training, the client's energy consumption is measured.
4. **Aggregate the local models**
	- The server combines the local models using FedAvg, weighting each client according to its number of local training examples:$$w^{r+1}=\sum_{i\in S_{\mathrm{sel}}}\frac{|D_i|}{|D_{\mathrm{total}}|}w_i^{r+1}$$
	 For example, if two clients have 80 and 20 training examples:$$w^{r+1}=0.8w_1^{r+1}+0.2w_2^{r+1}$$
5. **Evaluate and repeat**
	- Evaluate the new global model.
	- Increment the communication round $r$.
	- Repeat until the stopping condition is reached.

## Experimental Setup

### Dataset

- **Dataset:** CIFAR-10 image classification.
- The data is distributed among clients using a **Dirichlet distribution**: $\mathbf{p}\sim\operatorname{Dirichlet}(\beta,\ldots,\beta)$
	- $\beta$ controls the degree of data heterogeneity:
	  - $\beta=0.1$: highly non-IID and skewed.
	  - $\beta=0.5$: moderately non-IID.
	  - $\beta=1.0$: more homogeneous.
	Smaller $\beta$ values produce more uneven class distributions across clients.

### Model and Hyperparameters

- **Model:** LeNet CNN, approximately $0.24$ MB.
- **Input:** RGB images of size $3\times32\times32$.
- **Architecture:** 2 convolutional layers with max pooling and 3 fully connected layers.
- **Batch size:** 128.
- **Optimizer:** SGD.
- **Learning rate:** 0.01.
- **Momentum:** 0.5.

### Simulation Settings

- **Clients:** $N=50$.
- **Maximum rounds:** $R_{\max}=1500$.
- **Local epochs:** $L=5$.
- **Baselines:** FedAvg and FedProx, with random selection of $20\%$ of clients per round.
- **Temporary failures:** up to $30\%$ of selected clients may temporarily disconnect.
- **Permanent failures:** clients may permanently leave due to complete battery depletion or disconnection.
- **Recharging:** clients may recover a random amount of energy every 10 rounds.