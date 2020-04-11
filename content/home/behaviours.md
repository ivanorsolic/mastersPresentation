+++
weight = 9
+++
{{%section%}}
## Behavioural subnetwork

---

## The architecture
![](/images/behaviournet.png)

---

## Implementation

```python
# New input layer
behaviourInput = Input(shape=(numberOfBehaviourInputs,), name="behaviourInput")

# ConvNet parts ...

# Behavioural net
z = behaviourInput
z = Dense(numberOfBehaviourInputs * 2, activation='relu')(z)
z = Dense(numberOfBehaviourInputs * 2, activation='relu')(z)
z = Dense(numberOfBehaviourInputs * 2, activation='relu')(z)

# Concatenating the convolutional networks with the behavioural network
o = Concatenate(axis=1)([z, c])
o = Dense(100, activation='relu')(o)
o = Dense(50, activation='relu')(o)

# Output layers ...

# Update the model inputs
model = Model(inputs=[imageInput, laneInput, behaviourInput], outputs=[steering_out, throttle_out]) 
```

---

## Training

- 7k records with approx. 20 lane changes
- Final validation loss: **0.003947**

<img src="/images/trainingbehaviour.jpg" width="70%">

---

## Preview

<center><video controls src="/videos/changinglanes.mp4" autoplay muted loop width="100%"></video></center>

{{%/section%}}