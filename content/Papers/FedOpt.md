Paper: Adaptive Federated Optimization, Reddi et al., 2021

> [!IMPORTANT] Summary
> **FedOpt** is a framework that generalizes **FedAvg** by interpreting the aggregated client update as a **pseudo-gradient** and applying an adaptive optimizer (e.g., AdaGrad, Adam, or Yogi) on the **server** instead of a fixed SGD update. This simple modification leaves client training unchanged, preserves FedAvg’s theoretical convergence guarantees, and improves optimization robustness in practical federated learning scenarios.

## Overview

**FedAvg** is the de facto optimization algorithm for federated learning, but it suffers from several limitations:
- **The server update is overly simplistic.** After aggregating client models, the server simply averages the updates, equivalent to performing a single SGD step with a fixed learning rate ($\eta$=1). Unlike centralized training, it cannot exploit more advanced optimizers such as Adam or AdaGrad.
- **Federated updates are noisy and heterogeneous.** Clients train on different (often non-IID) datasets and perform multiple local SGD steps before communicating. As a result, the aggregated update received by the server is much noisier than the gradients encountered in centralized optimization.
- **FedAvg does not exploit information from previous communication rounds.** Each round is treated independently: once the server averages the client updates, all historical information is discarded. This prevents the server from adapting its optimization strategy based on past updates.

The key insight of **FedOpt** is that the aggregated client update behaves like a **pseudo-gradient** of the global objective.

Instead of directly averaging client updates as in FedAvg, the server treats the aggregated update as a gradient estimate and applies an adaptive optimizer (e.g., **AdaGrad**, **Adam**, or **Yogi**) to compute the next global model.

## Methods

<u>Assumptions</u>
- **Lipschitz Gradient**: $\|\nabla F_i(x)-\nabla F_i(y)\|\le L\|x-y\|$ → The gradient cannot change too abruptly.
- **Bounded Variance**: 
	- Local Variance: $\mathbb E\left[\|\nabla f_i(x,\xi)-\nabla F_i(x)\|^2\right] =\sigma_{l,i}^2$, where $f_{i(x,\xi)}$ is the loss on one example and $F_{i}(x)$ is the loss over all data → Noise between minibatches has bounded variance. This is the same assumption made in SDG.
	- Global Variance: $\frac1m\sum_i \|\nabla F_i(x)-\nabla F(x)\|^2 \le\sigma_g^2$. _This is unique to FL_, here, $\nabla F_i(x)-\nabla F(x)$ measures how different is client $i$’s direction from the global direction.
		- Small $\sigma_{g}$ means clients are similar (IID).
		- Large $\sigma_{g}$ means clients are heterogeneous (non-IID).
- **Bounded gradients**: $\|\nabla f_i(x,z)\|\le G$ → No stochastic gradient is allowed to become arbitrarily large.

<u>FedAvg Reformulation</u>
[[FedAvg]] simply averages clients updated weights: $$x_{t+1} = \frac1{|S|} \sum_{i\in S}x_t^i$$
Define $\Delta_t^i = x_t^i-x_t$ how much did client $i$ move away from the server model. Rewrite $x_{t+1} = x_t-\left(-\Delta_{t}\right)=x_{t}-g$ and we obtain:
$$x_{t+1} = x_t + \frac1{|S|} \sum_i\Delta_{t}^{i} = x_t+\Delta_{t}=x_{t}-\eta g$$
Now FedAvg looks like SDG and we can call $-\Delta_{t}$ a “_pseudo-gradient_” (because clients perform several local SGD steps and use stochastic gradients, but it’s close enough).

<u>FedOpt Main Idea</u>
FedAvg performs $x_{t+1} = x_t-\eta(-\Delta_t)$, which is simply SGD with learning rate 1.
==If the server already has something that looks like a gradient, why restrict the server to SGD?==

The server **never has access to the true global gradient** $\nabla F(x_t)$. Instead, it only receives the aggregated client update
$$  
\Delta_t=\frac{1}{|S|}\sum_{i\in S}\Delta_t^i,  
$$
which approximates a descent direction. Since
$$  
-\Delta_t\approx \nabla F(x_t),  
$$
the server can treat $-\Delta_t$ as a **pseudo-gradient** and apply **any gradient-based optimizer**, exactly as in centralized training.

Instead of the fixed FedAvg update
$$  
x_{t+1}=x_t+\Delta_t,  
$$
the server can maintain optimizer-specific state (e.g., momentum or adaptive learning rates) and perform
$$  
x_{t+1}=\text{SERVEROPT}(x_t,-\Delta_t).  
$$

<u>Implementation</u>
- **FedAdaGrad**
- **FedAdam**
- **FedYogi**

## Takeaway

FedOpt does **not** introduce a new client algorithm. It simply observes that the average client update behaves like a gradient and replaces the server’s fixed SGD update (FedAvg) with a more sophisticated optimizer such as AdaGrad, Adam or Yogi. This small change is sufficient to obtain consistent empirical improvements while preserving FedAvg’s convergence guarantees.