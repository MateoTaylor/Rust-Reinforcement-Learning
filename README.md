### Rust Reinforcement Learning

Main components of my Rust RL agent. Note this tests on any standard Rust environment, as it only intakes pixels from the screen.

I do augment training with game state data, which is collected via TCP with a couple custom C# plugins, (which I might upload later if it seems useful).

**Algorithm**
The model uses a standard PPO implementation, with cnn + LSTM backbone.

**Model**
- Resnet18 
    Uses all layers except final maxpool & classifier (I still maxpool later, but with my own sizings)
    Layers 1-2 are frozen, everything else is fine tuned along with the model. 
    
- feature classifying layers 
    Fully connected, outputs logits of certain game features existing on screen to lstm

- LSTM (inputs from conv and feature classifiying layers)

- standard PPO actor-critic policy heads (inputs from LSTM)

**Performance**
Thus far, agent is able to accomplish some very minimal tasks - avoiding water, walking towards resource nodes. I'm very compute limited at the moment, and don't expect to be able to train much without more time and a larger model (ideally 200m+ params). Current plan is to fiddle around with cloud computing until winter, when I'm gonna upgrade my PC.

