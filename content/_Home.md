## Definition

Federated Learning (FL) is a way of training machine learning models ==without collecting everyone’s data in one place==. Instead, the data stays on users’ devices or within organizations, and ==only small model updates are sent to a central server==. This helps protect privacy, reduces the need to move large amounts of data, and follows the principle of collecting only the information that is truly necessary. 

A major challenge in FL is the **federated optimization** setting, which differs from standard distributed optimization. Data is spread unevenly across clients: it is often **non-IID**, since each client reflects the behavior of a particular user or organization, and **unbalanced**, since some clients hold much more data than others. FL is also often **massively distributed**, with many more clients than examples per client, while communication is **limited** because devices may be offline or connected through slow or expensive networks. Because of these challenges, FL combines ideas from machine learning, distributed systems, cryptography, privacy, and statistics.

We can distinguish two main types of federated learning.
- **Cross-device FL** involves a very large number of personal devices, such as smartphones, where only a subset of devices participates in each training round.
- **Cross-silo FL** involves a smaller number of reliable organizations, such as hospitals or banks, collaborating to train a shared model while keeping their data local. In both cases, raw data never leaves the client; only model updates are exchanged and aggregated by a central server.

Companies such as Google and Apple use it in products like mobile keyboards and voice assistants, while organizations in healthcare, finance, and manufacturing are exploring cross-silo deployments.

## Index

### Foundations

From “*Kairuz et al., Advances and Open Problems in Federated Learning*”:
- [[Cross-Device Federated Learning Setting]]
- [[Non-IID Data in Federated Learning]]
- [[Communication and Compression]]
- [[Ensuring Fairness and Addressing Sources of Bias]]
- [ ] [[Addressing System Challenges]]

### Algorithms and Frameworks

- [[FedAvg]]
- [ ] FedAvgM
- [[FedOpt]]
- [[FedProx]]
- [[Oort]]
### Battery-Aware FL

- [[SAGE]]
- [[LeanFed]] (Possible Inconsistency)
- [[ESCS]]

### Compute- and System-Aware FL
