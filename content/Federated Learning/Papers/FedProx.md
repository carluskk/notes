Paper: _Federated Optimization in Heterogeneous Networks_, Li et al., 2020

> [!IMPORTANT] Summary
> FedProx extends FedAvg to better handle **statistical** (non-IID data) and **systems** (heterogeneous devices) heterogeneity in Federated Learning. Instead of minimizing only the local objective $F_k(w)$, each client solves $\min_w F_k(w) + \frac{\mu}{2}\|w-w_t\|^2,$ where the proximal term discourages the local model from drifting too far from the current global model. The server aggregation remains identical to FedAvg; only the client-side optimization changes. The paper’s main contribution is showing, both theoretically and experimentally, that this simple modification yields more stable convergence and better robustness when clients have heterogeneous data or perform different amounts of local computation.

## Overview

- Federated optimization algorithm proposed as an extension of **[[FedAvg]]** to address two major challenges commonly encountered in real-world Federated Learning:
	- **Statistical heterogeneity**, where clients hold data drawn from different (non-IID) distributions.
	- **Systems heterogeneity**, where clients have different computational capabilities, communication speeds, or available training time.
	Both forms of heterogeneity can significantly degrade the convergence of FedAvg.
- When clients have highly heterogeneous data or perform different amounts of local computation, their local models may drift significantly away from the current global model. This makes the aggregation step less stable and may slow down—or even prevent—convergence.

## Methods

==FedProx modifies only the local optimization problem solved by each client.==

In FedAvg, client k minimizes its local objective $\min_w F_k(w)$.

FedProx augments this objective with a proximal regularization term:

$$
\min_w F_k(w) + \frac{\mu}{2}\|w-w_t\|^2
$$

where

- $F_k(w)$ is the local loss function of client $k$;
- $w_t$ is the global model received from the server at the beginning of the communication round;
- $\mu$ is the proximal coefficient.

The additional term penalizes updates that move too far away from the current global model → **Soft Constraint**. Large deviations from the global model become increasingly expensive. As a result, ==local updates remain better aligned with the global optimization process while still capturing client-specific information==.

The gradient of the modified objective becomes:

$$
\nabla F_k(w) + \mu(w-w_t)
$$

The hyperparameter $\mu$ determines the strength of the proximal regularization:

- $\mu = 0$: FedProx reduces exactly to FedAvg.
- **Small $\mu$**: behavior remains very close to FedAvg.
- **Large $\mu$**: local models are constrained to stay closer to the global model, reducing local drift at the cost of less local adaptation.

The authors emphasize that **$\mu$ must be tuned empirically**, and the official implementation evaluates several values (e.g., 0.001, 0.01, 0.1, 0.5, and 1), as no single choice is optimal across all datasets.
## Takeaway

The primary goal of the paper is to provide a federated optimization framework that is robust to both **statistical heterogeneity** and **systems heterogeneity**. In particular, FedProx allows clients to perform **variable amounts of local work**, accommodating devices with different computational capabilities or intermittent availability. The proximal term stabilizes the optimization process under these heterogeneous conditions, making the algorithm significantly more robust than FedAvg in practical federated environments.
