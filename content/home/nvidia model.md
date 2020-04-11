+++
weight = 5
+++
#### DARPA Grand Challenge 2004
<section data-background-video="/videos/darpa.mp4" data-background-video-loop data-background-video-muted></section>

---

{{%section%}}

## First neural network

---

## Dave
##### (DARPA Autonomous Vehicle)

An RC Car with two cameras autonomously driving through a junk-filled alley way.

![](/images/dave.png)

---
## DAVE-2 Architecture

<img src="/images/daveArchitecture.png" height="550px">


{{%/section%}}

---

## Slight adjustions made:

{{% fragment %}} - ##### ✂ Omitted the normalization layer for now. {{% /fragment %}}

{{% fragment %}} - ##### ➕Added a 25 unit and a 5 unit layer. {{% /fragment %}}

{{% fragment %}} - ##### 💤 Added dropout regularization (90%). {{% /fragment %}}

{{% fragment %}} - ##### 🚙💨Two output units for steering and throttle. {{% /fragment %}}



---
{{%section%}}
## Implementing the DAVE-2 network

---

### Keras model

```python
def customArchitecture(num_outputs, input_shape, roi_crop):

    input_shape = adjust_input_shape(input_shape, roi_crop)
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    
    # Dropout rate
    keep_prob = 0.9
    rate = 1 - keep_prob
    
    # Convolutional Layer 1
    x = Convolution2D(filters=24, kernel_size=5, strides=(2, 2), input_shape = input_shape)(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 2
    x = Convolution2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 3
    x = Convolution2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 4
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 5
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(x)
    x = Dropout(rate)(x)

    # Flatten Layers
    x = Flatten()(x)

    # Fully Connected Layer 1
    x = Dense(100, activation='relu')(x)

    # Fully Connected Layer 2
    x = Dense(50, activation='relu')(x)

    # Fully Connected Layer 3
    x = Dense(25, activation='relu')(x)
    
    # Fully Connected Layer 4
    x = Dense(10, activation='relu')(x)
    
    # Fully Connected Layer 5
    x = Dense(5, activation='relu')(x)
    outputs = []
    
    for i in range(num_outputs):
        # Output layer
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=[img_in], outputs=outputs)
    
    return model
```

---

### Using the model in Donkey

```python
class NvidiaModel(KerasPilot):
    def __init__(self, num_outputs=2, input_shape=(120,160,3), roi_crop=(0,0), *args, **kwargs):
        super(NvidiaModel, self).__init__(*args, **kwargs)
        self.model = customArchitecture(num_outputs, input_shape, roi_crop)
        self.compile()

    def compile(self):
        self.model.compile(optimizer="adam",
                loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]
```

---

### Test drive
<center><video controls src="/videos/nvidia_architecture_test.mp4" autoplay muted loop width="100%"></video></center>

---
### Visualization
<center><video controls src="/videos/saliency.mp4" autoplay muted loop width="80%"></video></center>
{{%/section%}}




