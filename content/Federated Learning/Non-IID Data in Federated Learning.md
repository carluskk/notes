The most common sources of dependence and non-identicalness are due to each client corresponding to a particular **user**, a particular **geographic location**, and/or a particular **time window**.

Consider a supervised task with features $x$ and labels $y$. A statistical model of federated learning involves **two levels of sampling**: 
- Accessing a datapoint requires first sampling a client $i \sim Q$, the distribution over available clients;
- Drawing an example $(x, y) \sim P_{i}(x, y)$ from that client’s local data distribution.

When non-IID data in federated learning is referenced, this typically refers to differences between $P_{i}$ and $P_{j}$ for different clients $i$ and $j$. ==However, it is also important to note that the distribution $Q$ and $P_{i}$ may change over time, introducing another dimension of “non-IIDness”.==

From the optimization perspective, the global training objective is the average loss over all examples:

$$
f(w) = \frac{1}{n}\sum_{i=1}^{n} f_i(w).
$$

If the data is partitioned across $K$ clients, client $k$ has a local objective

$$
F_k(w) = \frac{1}{n_k}\sum_{i \in P_k} f_i(w),
$$

so the global objective can be rewritten as

$$
f(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w).
$$

In the IID case, each client is effectively a random sample of the whole population, so on average its local objective matches the global one: $\mathbb{E}[F_k(w)] = f(w)$. In federated learning this usually does not hold, so a client may optimize a local objective that differs significantly from the population objective. This is the optimization view of the non-IID setting.

## Characterizing Non-IID Data in Federated Learning

In federated learning, clients often have data generated from different distributions. Instead of simply stating that $P_i(x,y) \neq P_j(x,y)$ for two clients $i$ and $j$, we can decompose the joint distribution to understand *how* the data differs.

Using probability theory:

$$
P(x,y) = P(y|x)P(x) = P(x|y)P(y)
$$

This decomposition allows us to identify different sources of heterogeneity:
- **Feature Distribution Skew (Covariate Shift)**: $P_i(x) \neq P_j(x)$ while $P_i(y|x) = P_j(y|x)$. ==The inputs differ across clients, but the relationship between inputs and labels remains the same.==
  **Example**: Different users have different handwriting styles (slant, stroke width), but a given handwritten digit still corresponds to the same label.
- **Label Distribution Skew** $(P(y))$: The frequency of labels differs across clients, i.e., $P_i(y) \neq P_j(y)$, while $P_i(x|y) = P_j(x|y)$. ==The frequency of labels differs across clients, but each label looks the same everywhere.==
  **Example**: Certain emojis are much more common in some demographics or regions than others.
- **Same Label, Different Features (Concept Drift)**: $P_i(x|y) \neq P_j(x|y)$ while $P_i(y) = P_j(y)$. ==The meaning of a label is unchanged, but the features associated with that label vary across clients.==
  **Example**: The label “house” exists everywhere, but houses look very different in Italy, Japan, and the United States. **Same meaning, different appearance**.
- **Same Features, Different Labels (Concept Shift)**: $P_i(y|x) \neq P_j(y|x)$ while $P_i(x) = P_j(x)$. ==The same input may receive different labels depending on the client.== **Example:** The same movie review may be labeled as positive by one user and negative by another because of personal preferences. **Same appearance, different meaning**.
- **Quantity Skew (Unbalancedness)**: Clients hold different amounts of data, i.e., $|D_i| \neq |D_j|$, even if their data distributions are otherwise identical.

Real-world federated learning datasets likely contain a mixture of these effects, and the characterization of cross-client differences in real-world partitioned datasets is an important open question.
