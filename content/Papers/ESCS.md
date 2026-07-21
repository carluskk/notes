Paper: _Federated Learning Energy Saving Through Client Selection_, Maciel et al., 2024

> [!IMPORTANT] Summary
> ESCS is an energy-aware client-selection algorithm for Federated Learning that extends the authors' previous RAWCS approach. Before each communication round, clients report their execution context (battery level, training time, energy consumption, network quality, and evaluation loss). ESCS first filters clients that do not meet minimum battery and network requirements, then ranks the remaining candidates using either a system-based or model-based utility function. The algorithm supports both probabilistic and deterministic selection, resulting in four operating variants. Its objective is to reduce client energy consumption while maintaining the performance of the global model.

## Overview

- **Central Claim**: Existing client-selection methods mainly optimize statistical utility (e.g., convergence and model performance) or system aspects (e.g., training time and network conditions), while energy consumption has received comparatively less attention as a primary selection criterion.
- ESCS extends the authors' previous RAWCS algorithm by introducing a new **Energy Saving Client Selection (ESCS) algorithm** with four operating configurations.
- The goal of ESCS is to select clients by jointly considering **training time**, **battery level**, **energy consumption**, **model performance**, so as to reduce energy consumption while maintaining model performance.

## Methods

ESCS performs client selection in **two sequential subphases**:
1. **Filtering**: discard clients that do not satisfy minimum resource requirements.
2. **Picking**: rank the remaining eligible clients using a utility function and build the participant set.

Before each communication round, every available client sends its execution context `c_info` to the server. The reported information consists of:
- Evaluation loss (`c_l`)
- Battery level (`c_bl`)
- Model training energy consumption (`c_bc`)
- Training time (`c_tt`)
- Network quality (`c_nq`)

The server also defines a configuration `S_cfg` containing parameters such as:
- Minimum acceptable battery after a communication round (`B_LV_Lmin`)
- Maximum desired training time (`TT_max`)
- Minimum acceptable network quality (`NQ_min`)
- Utility weights (`α`, `β`, `γ`)
- Utility version (`C_uv`)
- Selection operation (`S_op`)

A client is accepted if  `(c_bl − c_bc) >= B_LV_Lmin` and `c_nq >= NQ_min`, otherwise the client is discarded. Once the eligible client set `S_ast` has been built, ESCS selects the final participant set `S_pt`. This is done through a utility function `U(c_info, S_cfg)` and the paper defines ==two variants==:
- **System-Based Utility**: The utility is built from the metrics “battery”, “training time” and “network quality”, with their importance controlled by the weights $\alpha, \beta, \gamma$. 
  N.B. Training time is not a hard constraint, but a penalization factor controlled by $\theta$.
- **Model-Based Utility**: Here: the **base** of the utility depends on **evaluation loss**, while the **exponent** depends on the system metrics. **IDEA**: “Among clients with similar statistical contribution, those with better battery, training time and network quality receive higher utility”.

**System-Based**:
$$
U(c_{info}, S_{cfg}) =
\left(
(\alpha \cdot c_{bl})
+
\left(\beta \cdot \left(1-\frac{c_{tt}}{Ctt_{max}}\right)\cdot\theta\right)
+
(\gamma \cdot c_{nq})
\right)^{\left(1-\frac{c_l}{Cl_{max}}\right)}
$$

**Model Based**:
$$
U(c_{info}, S_{cfg}) =
\left(
\frac{c_l}{Cl_{max}}
\right)^{
1-
\left(
(\alpha \cdot c_{bl})
+
\left(\beta \cdot \left(1-\frac{c_{tt}}{Ctt_{max}}\right)\right)
+
(\gamma \cdot c_{nq})
\right)
}
$$
### Algorithm

The standard FedAvg workflow is unchanged except that the random client selection is replaced by ESCS.

![[escs_algo.png | center]]

Algorithm 2 implements ESCS in two phases:
1. Filter candidate clients using R().
2. Select participants using U().

Two selection strategies are available:
- Probabilistic: each eligible client is selected with probability equal to its utility.
- Deterministic: eligible clients are ranked by utility and the top C fraction is selected.

The utility can be computed using either:
- System-based utility
- Model-based utility
## Takeaway

- **ESCS is a two-stage client-selection algorithm**: it first filters clients that do not satisfy minimum battery and network requirements, then selects participants based on a utility function.
- **Client utility can be computed in two ways**: a *System-based* utility, prioritizing battery level, training time, and network quality, or a *Model-based* utility, prioritizing evaluation loss while still accounting for system conditions.
- **ESCS supports probabilistic and deterministic selection**, resulting in four variants (ESCS-SP, ESCS-SD, ESCS-MP, ESCS-MD) depending on the utility function and selection strategy.
- **The goal of ESCS is to reduce client energy consumption while preserving model performance**, extending the authors' previous RAWCS algorithm with a new energy-aware selection strategy and an improved energy simulation model.