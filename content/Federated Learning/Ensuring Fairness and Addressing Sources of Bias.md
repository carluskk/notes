Machine learning models can often exhibit surprising and unintended behaviors:
- **Individual Fairness**: If people with similar characteristics receive quite different outcomes.
- **Demographic Fairness**: If certain sensitive groups (races, genders, etc.) receive different patterns of outcomes, e.g., different false negative rates.
- **Counterfactual Fairness**: Requires that a user receive the same treatment as they would have if they had been a member of a different group (race, gender, etc), after taking all causally relevant pathways into account.
## Bias in Training Data

One common anti-pattern is that minority or marginalized social groups are under-represented in the training data, and thus the learner weights these groups less during training, leading to inferior quality predictions for members of these groups.

Examples of biases introduced in federated learning:
- If devices are selected for updates when plugged-in or fully charged → Correlation with factors such as day-shift vs night-shift work schedules.
- If devices are selected for updates from among the pool of eligible devices at a given time.
- If selected devices are more likely to have their output kept when the output is computed faster:
	- Output from devices with faster processors may be over-represented → Correlation with socioeconomic status.
	- Devices with less data may be over-represented → Over-representation of users who use the product less frequently.
- If the update frequency depends on latency → Under-representation of users in certain geographic regions and/or with slower devices or networks.

Biases in the data-generating process can also drive unfairness in the resulting models learned from this data. Maybe **the collected interaction data may encode users’ workarounds for a poorly designed product rather than their true preferences or intentions**.
Example: Autocompletion for bilingual users.

## Fairness Without Access to Sensitive Attributes

**How can we measure and improve fairness if we don’t know who belongs to which demographic group?**

The idea is that, instead of protecting known groups, we protect whoever is doing worst – **Distributionally-Robust Optimization (DRO)**. DRO tries to optimize the worst-case performance, rather than the average one. Make sure nobody is doing terribly, even if we don’t know who they are.

For **Multicalibration**, on the other hand, we make sure predictions are reliable across many different subsets of users.

> For the authors of “*Advances and Open Problems in Federated Learning*” we should reframe the problem, shifting the focus from having equal outcomes to **equal access to a good model**. Under this interpretation of fairness, the goal is to maximize model utility across all individuals, regardless of their (unknown) demographic identities, and regardless of the “goodness“ of an individual outcome.

## Fairness, Privacy, and Robustness

> **Remember**:
> Privacy = Aims to protect the model from being too dependent on individuals.
> Fairness = Encourages the model to perform well on under-represented classes.

**Can a model be private, fair, and robust at the same time?** MEH

Differentially-private learning typically seeks to obscure individually identifying characteristics, while fairness often requires knowing individuals’ membership in sensitive groups in order to measure or ensure fair predictions are being made.

A potential solution to relax the tension between privacy and fairness may be **personalization** and “**hybrid differential privacy**”, i.e., where some users donate data with lesser privacy guarantees.

**N.B.** ==Empirical studies have shown evidence that differentially-private optimization impacts minority subgroups the most.==

*“Modifications to differentially private optimization algorithms which explicitly seek to preserve performance on minority subgroups, e.g. by ==adapting the noise and clipping mechanisms to account for the representation of groups within the data==, would also likely do a great deal to limit potential disparate impacts of differentially-private modeling on minority subgroups in federated models trained with differential privacy. However, ==implementing such adaptive differentially-private mechanisms in a way that provides some form of privacy guarantee presents both algorithmic and theoretical challenges== which need to be addressed by future work.”*

## Leveraging Federation to Improve Model Diversity

Federated learning presents an opportunity to leverage uniquely diverse datasets by providing efficient decentralized training protocols along with privacy and non-identifiability guarantees for the resulting models. ==This provides a practical opportunity to leverage larger, more diverse datasets and explore the generalizability of models which were previously limited to small populations.==
