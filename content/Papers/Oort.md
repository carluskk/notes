Paper: _Oort: Efficient Federated Learning via Guided Participant Selection_, Lai et al., 2021

> [!IMPORTANT] Summary
> Oort is a guided participant selection framework for Federated Learning that improves time-to-accuracy by selecting clients based on both their statistical utility (training loss) and system performance (training speed). It uses an exploration-exploitation strategy to favor known high-utility clients while continuously discovering and reconsidering others. For federated testing, Oort selects enough clients to obtain representative results or, when client data distributions are known, matches a target data distribution while minimizing testing time.

## Overview
- Oort is a **participant selection framework integrated into the FL coordinator**. It selects valuable clients according to developer-defined criteria and execution feedback, while the FL execution framework remains responsible for running the actual training.
- Oort improves time-to-accuracy performance by $1.2\times-14.1\times$ and final model accuracy by $1.3\%-9.8\%$
## Architecture and Interface

## Architecture Overview

![[oort_1.png| center | 700]]

1. **Job submission:** The developer defines the participant selection criteria.
2. **Participant selection:** The coordinator identifies eligible clients and provides their characteristics to Oort. Oort selects the participants for the current round.
3. **Execution:** The coordinator sends the global model to the selected clients, which perform local training on their data.
4. **Aggregation:** The coordinator aggregates the clients' model updates.

During federated training, the **selection → execution → aggregation** process is repeated at every communication round.

Oort employs two distinct selectors that developers can access via a client library during FL training and testing. (1) Keep track the utility of clients in training (2) Efficiently explores and selects high-utility clients at runtime.

### Training Selector

At each round, the FL execution engine returns feedback from the previous round, i.e. how useful a client’s training result was, how long the client took to finish or if it successfully completed the execution.

**Main Idea**: favor clients that are both:
- Statistically useful for model training;
- Efficient enough not to slow down the round too much.

```python
import Oort


def federated_model_training():
    # Create the Oort training selector
    selector = Oort.create_training_selector(config)

    # Train until the target testing accuracy is reached
    while federated_model.testing() < target:

        # Train for 50 communication rounds before testing again
        for _ in range(50):

            # Collect feedback from participants in the previous round
            feedbacks = engine.get_participant_feedback()

            # Update Oort's utility estimate for each client
            for client_id in feedbacks:
                selector.update_client_util(
                    client_id,
                    feedbacks[client_id]
                )

            # Select 100 high-utility participants
            participants = selector.select_participant(100)

            # Activate training on the selected remote clients
            # ...
```

<u>Client Statistical Utility</u>

Oort estimates how **useful a client's local data is for improving the global model**. The theoretically optimal importance-sampling approach relies on the $L_2$-norm of per-sample gradients, but computing these gradients for every local sample is impractical.

Oort therefore approximates the statistical utility of client $i$ using the training loss already computed during local training:
$$
U(i)=|B_i|\sqrt{\frac{1}{|B_i|}\sum_{k\in B_i}\text{Loss}(k)^2}
$$
where $B_i$ is the local dataset of client $i$. Clients with **more data and higher accumulated training loss** receive higher statistical utility, since the current model is expected to have more to learn from their data.

<u>Balance Statistical and System Efficiency</u>

Oort combines a client's **statistical utility** with its **system efficiency** to avoid selecting statistically valuable clients that significantly slow down training:
$$
\text{Util}(i)=U(i)\left(\frac{T}{t_i}\right)^{\mathbb{1}(T<t_i)\alpha}
$$
where $T$ is the desired round duration, $t_i$ is the training time of client $i$, and $\alpha$ controls the penalty applied to slow clients. Clients with $t_i \leq T$ receive no penalty, while slower clients are penalized unless their statistical utility is high enough to compensate for their execution time.

<u>Participant Selection: Exploration and Exploitation</u>

Since a client's utility is only known after participation and may become outdated as training progresses, Oort models participant selection as a **multi-armed bandit problem**, where each client is an arm and its observed utility is the reward. At each round, Oort balances **exploitation of known high-utility clients** with **exploration of unseen clients**.

- **Exploitation:** approximately $(1-\varepsilon)K$ participants are selected from known high-utility clients. Oort uses a confidence parameter $c$ to include clients whose utility is at least $c$ times the utility at the selection cut-off, creating a high-utility candidate pool. Clients are then sampled from this pool with probability proportional to their utility.
- **Exploration:** the remaining $\varepsilon K$ participants are selected from previously unexplored clients. When system information is available, Oort prioritizes faster unexplored devices rather than exploring randomly.
- **Staleness:** clients that have not participated for a long time receive an increasing utility incentive, giving clients with outdated utility estimates a chance to be reconsidered.
- **Robustness:** Oort limits repeated selection of the same clients and clips extreme utility values, reducing the influence of corrupted or noisy clients with abnormally high training loss.

Oort can also incorporate a developer-defined fairness criterion into client utility:
$$
(1-f)\text{Util}(i)+f\cdot\text{fairness}(i),
\qquad f\in[0,1]
$$
where $f$ controls the trade-off between **time-to-accuracy efficiency** and **fairness**.
## Testing Selector

Chooses clients that make the evaluation set representative while trying to finish testing quickly:
- If client data characteristics are unknown, Oort estimates how many participants are required to keep the selected test data distribution sufficiently close to the global distribution.
- If client data characteristics are known, Oort selects clients whose combined data satisfy the developer-specified test distribution while minimizing testing duration.

![[oort_2.png| center | 600]]

Oort optimizes **time-to-accuracy** by jointly considering **statistical efficiency** (number of rounds required to reach the target accuracy) and **system efficiency** (duration of each training round). Selecting only statistically useful clients may introduce slow devices, while selecting only fast clients may provide redundant data and require more rounds.

If client data characteristics are unknown, Oort defines the deviation between the selected test data and the true global distribution using the $L_1$-distance:
$$\|\bar X-E[\bar X]\|_1$$
The developer specifies a maximum acceptable deviation $\epsilon$ and a confidence level $\delta$. Then Oort estimates the **minimum number of clients** m needed such that: $$\Pr \left[ \|\bar X-E[\bar X]\|_1\leq\epsilon \right] \geq\delta$$
> “How many random clients do I need so that the resulting test data is within $\epsilon$ of the global distribution with, say, 95% confidence?”

If, on the other hand, the client data characteristics are known, Oort can explicitly choose clients whose combined data satisfy this target distribution and minimizes the maximum client completion time: $$\min \max_{j}\left(\sum_{i} c_{i} n_{i}^{j}s_{j}+\frac{d_{j}}{b_{j}}\right) = \min\text{(Time of the last client to finish)}$$
where:
- $c_{i}$ is the computation cost of processing one sample category $i$.
- $n_{i}^{j}$ is the number of samples of category $i$ contributed by client $j$.
- $s_{n}$ is the computation speed of client $n$.
- $d_{n}$ is the amount of data to communicate.
- $b_{n}$ is the available bandwidth.
## Takeaway

Oort is a **guided participant selection framework** designed to improve FL **time-to-accuracy**. During training, it prioritizes clients with high statistical utility while penalizing slow devices, and uses an **exploration-exploitation strategy** to discover new clients and handle stale utility estimates. Client utility is approximated using local training loss, avoiding expensive per-sample gradient computations. For federated testing, Oort either estimates the number of clients required for a representative evaluation or selects clients whose combined data match a target distribution. Overall, Oort treats **participant selection as a joint statistical and system optimization problem**, rather than randomly selecting clients.
