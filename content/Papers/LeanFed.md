Paper: _Learn More by Using Less: Distributed Learning with Energy-Constrained Devices_, Pereira et al., 2025

> [!IMPORTANT] Summary
> Energy-aware federated learning method that **reduces the fraction of local data processed by energy-constrained devices to prolong their participation in training**. Based on each device's battery budget, training cost, and expected participation rate, it computes a data fraction $\eta_e$ and trains locally only on that subset before applying standard FedAvg aggregation. The core goal is to prevent clients from exhausting their batteries early, since losing clients, especially under non-IID data, can cause the global model to overfit to the few remaining devices and degrade in accuracy.

## Overview

- Energy-aware FL framework designed to optimize CS and training workloads on battery-constrained devices.
- Idea: Dynamically adjust the fraction of local data each device utilizes during training → Maximize device participation ensuring no battery run-out.
- Focus: Scenarios where multiple devices may lack sufficient battery life to participate in all communication rounds $R$ pre-defined by the central server.

## Methods

For each device $e$, the maximum number of full-data communication rounds it can support is estimated as: $$R_e = \frac{B_e}{\bar{b}_e L_e}$$where $B_e$ is the device's energy budget, $\bar{b}_e$ is its estimated training energy cost, and $L_e$ is the number of local epochs.

If the device cannot naturally sustain the required $R$ communication rounds ($R_e \leq R$), LeanFed computes a local data fraction $\eta_e \in [0,1]$: $$\eta_e(\lambda)=R\frac{\bar{b}_eL_e}{\lambda B_e}$$where $\lambda$ is a server-defined scaling factor accounting for the expected device participation. The device trains only on the corresponding percentage of its local dataset, reducing computation and energy consumption.

> [!DANGER]
> **Possible inconsistency in Eq. (4).** Since $B_e$ appears in the denominator and $\eta_e$ is recomputed at each round, a decrease in residual battery leads to an increase in $\eta_e$. This seems opposite to the stated goal of reducing local computation for energy-constrained devices and may indicate a typo or an unclear interpretation of the equation.
### Algorithm

1. **Global model broadcast:** At each communication round, the server sends the current global model to all devices.
2. **Data fraction estimation:** Each active device computes the fraction $\eta_e$ of its local dataset to use.
3. **Local training:** The device performs $L_e$ local epochs using only the $\eta_e$ fraction of its local dataset.
4. **Energy update:** The energy consumed during local training is subtracted from the device's remaining battery $B_e$.
5. **Model transmission:** Devices with remaining energy send their updated local models to the server.
6. **Aggregation:** The server aggregates the models of the active devices using FedAvg and updates the global model.
7. **Repeat:** The process continues for $R$ communication rounds.
## Takeaway

LeanFed adjusts the fraction of local data processed by each device according to its energy constraints, aiming to reduce per-round energy consumption and keep devices active throughout federated training.