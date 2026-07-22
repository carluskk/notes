![[cross_device_map.png|center]]

A server (service provider) orchestrates the training process, by repeating the following steps until training is stopped:
1. **Client selection**: The server samples from a set of clients meeting eligibility requirements. For example, mobile phones might only check in to the server if they are plugged in, on an unmetered Wi-Fi connection, and idle, in order to avoid impacting the user of the device.
2. **Broadcast**: The selected clients download the current model weights and a training program.
3. **Client computation**: Each selected device locally computes an update to the model by executing the training program, which might for example run SGD on the local data (as in Federated Averaging).
4. **Aggregation**: The server collects an aggregate of the device updates. For efficiency, stragglers might be dropped at this point once a sufficient number of devices have reported results.
   ==**N.B.** This stage is also the integration point for many other techniques, including secure aggregation for added privacy, lossy compression of aggregates for communication efficiency, and noise addition and update clipping for differential privacy.==
5. **Model update**: The server locally updates the shared model based on the aggregated update computed  from the clients that participated in the current round.
 
![[cross_device_numbers.png|center|600]]
**Order-of-magnitude sizes for typical cross-device federated learning applications.**

The separation of the client computation, aggregation, and model update phases is not a strict requirement of federated learning, and it indeed excludes certain classes of algorithms, for example asynchronous SGD where each client’s update is immediately applied to the model, before any aggregation with updates from other clients.

## Synchronous vs Asynchronous Federated Learning

In synchronous FL, the server sends the current global model to a set of selected clients. Each client trains locally on its own data and computes a model update. ==The server waits until enough clients have finished, aggregates their updates (typically by averaging), and then performs a single update to the global model.== All participating clients therefore contribute to the same training round.

In asynchronous FL, the server does not wait for multiple clients before updating the model. ==As soon as a client finishes local training, its update is immediately incorporated into the current global model.== The server continuously updates the model as client updates arrive, rather than first aggregating them into a single update.

The main difference is when the global model is updated:
- **Synchronous FL:** collect many client updates → aggregate → perform one global update.
- **Asynchronous FL:** receive one client update → immediately update the global model.

Asynchronous methods can reduce waiting time and improve system efficiency, but they introduce the problem of **stale updates**, since some clients may compute their updates using an older version of the global model. Synchronous FL avoids this issue and naturally supports techniques such as secure aggregation, differential privacy, and communication-efficient compression.