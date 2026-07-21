Paper: _Communication-Efficient Learning of Deep Networks from Decentralized Data_, McMahan et al., 2023

> [!IMPORTANT] Summary
> **FederatedAveraging (FedAvg)** trains a global model by repeatedly: (1) sampling a fraction $C$ of clients, (2) sending them the current global model $w_t$, (3) letting each selected client run local SGD for $E$ epochs with minibatch size $B$, and (4) averaging the returned client models with weights proportional to the number of local examples. FedAvg generalizes FedSGD: the special case $E=1$ and $B=\infty$ is exactly FederatedSGD.

## Overview

FedAvg is motivated by the fact that in federated learning, **communication is much more expensive than local computation**. Instead of communicating after every small update, each selected client performs several local optimization steps before sending its result back to the server. The server then aggregates these locally trained models by a weighted average.

This makes each communication round more informative and can drastically reduce the total number of communication rounds needed to train a model.

## Methods

### From the Global Objective to FedSGD

Let the global objective be

$$
f(w) = \frac{1}{n}\sum_{i=1}^{n} f_i(w),
$$

where:
- $w \in \mathbb{R}^d$ is the model parameter vector,
- $n$ is the total number of training examples,
- $f_i(w)$ is the loss associated with example $i$.

Assume the data is partitioned across $K$ clients. Let:
- $P_k$ be the set of indices of examples stored on client $k$,
- $n_k = |P_k|$ be the number of examples on client $k$,
- $\sum_{k=1}^{K} n_k = n$.

The local objective of client $k$ is
$$
F_k(w) = \frac{1}{n_k}\sum_{i \in P_k} f_i(w).
$$
The global objective can then be rewritten as
$$
f(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w).
$$
The baseline algorithm considered in the paper is **FederatedSGD (FedSGD)**. At communication round $t$, client $k$ computes the gradient of its local objective at the current global model $w_t$:
$$
g_k = \nabla F_k(w_t).
$$
The server aggregates these gradients and updates the global model as
$$
w_{t+1} \leftarrow w_t - \eta \sum_{k=1}^{K} \frac{n_k}{n} g_k,
$$
where $\eta$ is the learning rate. Since
$$
\sum_{k=1}^{K} \frac{n_k}{n} g_k = \nabla f(w_t),
$$
this is exactly a gradient descent step on the global objective.

<u>Equivalent Model-Averaging View</u>

The same FedSGD update can be rewritten in an equivalent form. Each client first takes one local gradient step:
$$
w_{t+1}^k \leftarrow w_t - \eta g_k,
$$
and then the server averages the resulting client models:
$$
w_{t+1} \leftarrow \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^k.
$$
This equivalence is the key step that leads to FedAvg: once the algorithm is written as **local training followed by model averaging**, each client can perform more than one local update before the averaging step.

### Algorithm

FedAvg extends FedSGD by allowing each selected client to run multiple local SGD updates before communicating with the server.

The algorithm is controlled by three main parameters:
- $C$: fraction of clients selected on each round,
- $E$: number of local epochs run by each selected client,
- $B$: local minibatch size used by each client.

If there are $K$ total clients, then the number of selected clients per round is
$$
m = \max(CK, 1).
$$
For a client with $n_k$ local examples, the number of local updates per round is
$$
u_k = E \frac{n_k}{B}.
$$
The special notation $B = \infty$ means the entire local dataset is treated as one batch.

Therefore, the special case $E = 1, B = \infty$ reduces exactly to FedSGD.

<u>Pseudocode Structure</u>

At round $t$:
1. The server samples a subset $S_t$ of $m$ clients.
2. The server sends the current global model $w_t$ to each client in $S_t$.
3. Each selected client runs local SGD starting from $w_t$ and returns an updated local model $w_{t+1}^k$.
4. The server computes a weighted average of the returned models.

Let
$$
m_t = \sum_{k \in S_t} n_k
$$
be the total number of local examples across the participating clients in round $t$. The server update is then
$$
w_{t+1} \leftarrow \sum_{k \in S_t} \frac{n_k}{m_t} w_{t+1}^k.
$$
Thus, clients with more local data contribute proportionally more to the next global model.

<u>ClientUpdate</u>

On client $k$, starting from the received global model $w$, the paper’s `ClientUpdate` routine is:

1. Split the local dataset $P_k$ into minibatches of size $B$.
2. For each local epoch $i = 1, \dots, E$:
3. For each minibatch $b$:

$$
w \leftarrow w - \eta \, \nabla \ell(w; b),
$$

where $\ell(w; b)$ is the minibatch loss on batch $b$.

After all local SGD steps, the client returns the final local model to the server.

## Takeaway

### Why Model Averaging Is Not Trivial

For general **non-convex** objectives, averaging models in parameter space can in principle produce an arbitrarily bad model. Neural networks are not convex, so two parameter vectors that each correspond to good solutions may have an average that performs poorly.

The paper illustrates this with MNIST: if two digit-recognition models are trained independently from **different random initializations** on two non-overlapping IID subsets, then naively averaging their parameters can behave badly.

More precisely, in the experiment described in the paper:
- each model is trained on a separate IID sample of 600 MNIST examples,
- training uses SGD with learning rate $0.1$,
- minibatch size is $50$,
- the models are trained for $240$ updates,
- this corresponds to about $E=20$ passes over each local dataset.

This is roughly the point at which the models begin to overfit their own local datasets, and parameter averaging across independently initialized models performs poorly.

### Why FedAvg Works

The paper then gives the practical intuition for why FedAvg still works well.

If two models start from the **same initialization** and are then trained independently on different subsets of the data, parameter averaging works much better. In that case, the averaged model can even achieve lower loss on the full training set than either model trained in isolation.

This is directly relevant to FedAvg, because at each round all participating clients start from the same global model $w_t$. Therefore, their local models remain relatively close in parameter space, making weighted averaging much more reasonable than averaging two completely unrelated neural networks.

### Why Averaging Works Despite Non-Convexity

FedAvg does not rely on convexity of the loss. If the objective were convex, then for any two parameter vectors $w$ and $w'$ and any $0 \leq \lambda \leq 1$,

$$
f(\lambda w + (1-\lambda)w') \leq \lambda f(w) + (1-\lambda)f(w').
$$

In particular, at the midpoint we would have

$$
f\left(\frac{w+w'}{2}\right) \leq \frac{f(w) + f(w')}{2},
$$

so parameter averaging would be naturally safe. Neural network objectives are generally non-convex, so this guarantee does not hold: two independently trained models may lie in different regions of parameter space, and their average may correspond to a high-loss model.

FedAvg works in practice for a different reason. At round $t$, all participating clients start from the same global model $w_t$ and perform only limited local training. Therefore, each returned client model can be written as

$$
w_{t+1}^k = w_t + \Delta_k,
$$

where $\Delta_k$ is the update computed locally by client $k$. The server then aggregates

$$
w_{t+1} = \sum_{k \in S_t} \frac{n_k}{m_t} w_{t+1}^k
= w_t + \sum_{k \in S_t} \frac{n_k}{m_t}\Delta_k.
$$

So FedAvg is averaging nearby perturbations of the same starting point, rather than averaging arbitrary independently trained networks. This makes model averaging much more reasonable in practice, although it is still not guaranteed to work in full generality.

### Practical Interpretation

FedAvg can be viewed as a communication-efficient variant of distributed SGD:
- clients do several local SGD steps,
- communication happens only after this local work,
- the server aggregates local models by a weighted average.

Its strength is exactly the tradeoff it exploits:

$$
\text{more local computation} \quad \Longrightarrow \quad \text{fewer communication rounds.}
$$

This is why FedAvg became the canonical baseline algorithm in federated learning.

> [!DANGER] Watch Out
> FedAvg becomes more communication-efficient as local computation increases, but this also increases the risk that a client drifts toward optimizing its local objective $F_k$ rather than the global objective $f$. This tension becomes especially important in the **non-IID** setting, where different clients may have very different local data distributions.

## Empirical Observations

**Core empirical lesson**: FedAvg beats FedSGD mainly because each communication round does more useful local work. This greatly reduces communication, but too much local work can cause client drift, especially on harder non-IID tasks.

What to remember:
- Use $C=0.1$ as a reasonable default client fraction; more clients smooth training, but gains eventually diminish.
- Communication savings mostly come from increasing local work:

$$
u = \frac{\mathbb{E}[n_k]}{B}E = \frac{nE}{KB}.
$$

Larger $u$ usually means fewer communication rounds.
- Increase local work by raising $E$, lowering $B$, or both. In practice, tune $B$ first because smaller minibatches can add local updates without much extra cost if hardware is still used efficiently.
- FedAvg works well even with non-IID data, including extreme MNIST splits and realistic Shakespeare data.
- But very large $E$ can plateau or diverge, especially for language models. More local computation is useful only up to the point where client drift becomes too strong.
- FedAvg can also reach better final accuracy than FedSGD, possibly because model averaging has a regularizing effect.

**Practical rule**: Start with moderate client participation and local work, then increase local computation until communication improves without instability. Later in training, reducing local work can help, similar to learning-rate decay.
