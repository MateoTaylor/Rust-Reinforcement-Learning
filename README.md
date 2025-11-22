### Rust Reinforcement Learning

Main components of my Rust RL agent. Note this tests on any standard Rust environment, as it only intakes pixels from the screen.
However, it does train by communicating via TCP with a couple custom C# plugins, (which I might upload later if it seems useful).

The model uses a standard PPO implementation, along with some added cross entropy loss for the aux feature classifying layers.

Model consists of
- 4 conv layers (2-4 are batch normed, max pool after 4).
- feature classifying layers (3 fully connected, outputs logits of certain game features existing on screen to lstm)
- LSTM (inputs from conv and feature classifiying layers)
- standard linear actor-critic policy heads (from LSTM)

