In federated optimization, communication is usually the main bottleneck, while local computation on clients is relatively cheap. For this reason, it is often beneficial to trade extra computation for fewer communication rounds. This can be done by increasing the number of participating clients per round or by increasing the amount of local computation each client performs before communicating. In practice, once a sufficient level of client parallelism is reached, the largest gains often come from increasing local computation.

Motivated by the limited resources of current devices in terms of compute, memory and communication, there are several different compression objectives of practical value.

- **Gradient compression**: Reduce the size of the object communicated from clients to server, which is used to update the global model.
- **Model broadcast compression**: Reduce the size of the model broadcast from server to clients, from which the clients start local training.
- **Local computation reduction**: Any modification to the overall training algorithm such that the local  training procedure is computationally more efficient.

==Gradient compression refers to reducing the size of the client-to-server model update (gradient or local model update) before transmission, using techniques such as quantization, sparsification, or top-k selection, in order to reduce communication costs in federated learning.==

Very few methods try to solve (a), (b), and (c) simultaneously. In practice, a perfect FL system would send less data up, send less data down and require less local computation.

A key distinction exists between **cross-device** and **cross-silo** federated learning. In cross-device FL, clients are unreliable and may not participate again in future rounds, making it difficult to use compression methods that rely on storing information locally across rounds. In contrast, cross-silo FL involves persistent participants (e.g., hospitals or organizations), allowing the use of error-correction and stateful compression techniques.

Another important research direction is the development of methods that produce a **compact final model** while simultaneously reducing communication and computation costs during training. Achieving all of these goals together remains an open challenge.

Finally, current gradient compression methods are often analyzed using **worst-case guarantees**, meaning they provide a fixed compression benefit regardless of the structure of the updates. The authors argue that future methods should be **data-dependent**, adapting the compression strategy to the actual gradients or model updates being transmitted, similarly to how classical data compression exploits redundancy in the data being compressed.

Federated learning systems often rely on additional privacy-preserving mechanisms beyond keeping data local:
- **Secure Aggregation** is a cryptographic protocol that allows the server to compute only the aggregate of client updates (e.g., their sum or average) without observing any individual update.
- **Differential Privacy (DP)** protects users by adding carefully calibrated noise to model updates, limiting the amount of information that can be inferred about any single participant. However, integrating these techniques with communication-efficient compression methods is challenging. 

Secure Aggregation introduces additional communication overhead, which can significantly reduce the benefits of aggressive update compression. Similarly, most differential privacy mechanisms assume real-valued updates and the addition of continuous noise, while many compression techniques rely on quantization and discrete representations, making the two approaches difficult to combine. As a result, designing compression schemes that remain efficient while preserving both Secure Aggregation and Differential Privacy guarantees remains an important open research problem in federated learning.
