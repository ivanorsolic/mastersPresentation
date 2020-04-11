+++
weight = 8
+++
{{%section%}}

## Lane extraction

---

## Mapping the 3D world to 2D

<img src="/images/perspective1.jpg" width="80%">

---

## Parallel?

<img src="/images/perspective2.png" width="80%">

---

## Another example

<img src="/images/perspective3.jpg" width="80%">

---

## Parallel?

<img src="/images/perspective4.jpg" width="80%">

---

## Train dronez?

<img src="/images/perspective5.jpg" width="45%">

---

## Warping the perspective

<center><img src="/images/warped2.png" width="50%"><img src="/images/warped.png" width="50%"></center>

---

## First step: ROI

<img src="/images/perspective7.jpg" width="80%">

---

## Second step: target perspective

<img src="/images/perspective8.jpg" width="80%">

---

## Final step: transform

<img src="/images/hsl1.jpg" width="80%">

---

## Making the lines easier to see

---

## Converting to HSL color space
<img src="/images/hsl1.png" width="80%">

---

## Split into separate channels
<center><img src="/images/hsl2.png" width="33%"><img src="/images/hsl4.png" width="33%"><img src="/images/hsl3.png" width="33%"></center>

---

## Threshold the S-channel
<img src="/images/hsl5.png" width="80%">

---

## The entire process
<center><video controls src="/videos/lanelines.mp4" autoplay muted loop width="100%"></video></center>

---

## Going one step further?

---

### Fitting lanes using a polynomial approximation
<center><img src="/images/hsl5.png" width="45%"> &nbsp;<img src="/images/histogram.png" width="45%"></center>

---

## Histogram overlay
<img src="/images/overlay.png" width="80%">

---

### Found lane using polynomial fit
<center><img src="/images/preview4.jpg" width="80%"></center>

---

### Unwarped image with detected lane
<center><img src="/images/preview6.jpg" width="80%"></center>

---
## Polynomial fit in action
<center><video controls src="/videos/lanefinding.mp4" autoplay muted loop width="100%"></video></center>

---

## Why not to use it?

{{%/section%}}

---

### Implementing the lane finding

```python
def warpImage(self, image):
    # Define the region of the image we're interested in transforming
    regionOfInterest = np.float32(
    [[0,  180],  # Bottom left
    [112.5, 87.5], # Top left
    [200, 87.5], # Top right
    [307.5, 180]]) # Bottom right

    # Define the destination coordinates for the perspective transform
    newPerspective = np.float32(
    [[80,  180],  # Bottom left
    [80,    0.25],  # Top left
    [230,   0.25],  # Top right
    [230, 180]]) # Bottom right
    # Compute the matrix that transforms the perspective
    transformMatrix = cv2.getPerspectiveTransform(regionOfInterest, newPerspective)
    # Warp the perspective - image.shape[:2] takes the height, width, [::-1] inverses it to width, height
    warpedImage = cv2.warpPerspective(image, transformMatrix, image.shape[:2][::-1], flags=cv2.INTER_LINEAR)
    return warpedImage
  
def extractLaneLinesFromSChannel(self, warpedImage):
    # Convert to HSL
    hslImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2HLS)
    # Split the image into three variables by the channels
    hChannel, lChannel, sChannel = cv2.split(hslImage)
    # Threshold the S channel image to select only the lines
    lowerThreshold = 65
    higherThreshold = 255
    # Threshold the image, keeping only the pixels/values that are between lower and higher threshold
    returnValue, binaryThresholdedImage = cv2.threshold(sChannel,lowerThreshold,higherThreshold,cv2.THRESH_BINARY)
    # Since this is a binary image, we'll convert it to a 3-channel image so OpenCV can use it
    thresholdedImage = cv2.cvtColor(binaryThresholdedImage, cv2.COLOR_GRAY2RGB)
    return thresholdedImage

def processImage(self, image): 
    warpedImage = self.warpImage(image)
    # We'll normalize it just to make sure it's between 0-255 before thresholding
    warpedImage = cv2.normalize(warpedImage,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    thresholdedImage = self.extractLaneLinesFromSChannel(warpedImage)
    one_byte_scale = 1.0 / 255.0 
    # To make sure it's between 0-1 for the model
    return np.array(thresholdedImage).astype(np.float32) * one_byte_scale
```

---

{{%section%}}
## Implementing the first iteration model
<center><img src="/images/nnWithoutBehavior.png" width="100%"></center>

---
## 	Model definition

```python
def oriModel(inputShape, numberOfBehaviourInputs):

    # Dropout rate
    keep_prob = 0.9
    rate = 1 - keep_prob
    
    # Input layers
    imageInput = Input(shape=inputShape, name='imageInput')
    laneInput = Input(shape=inputShape, name='laneInput')
```

---

## Generalized CNN

```python
# Input image convnet
    x = imageInput
    x = Conv2D(24, (5,5), strides=(2,2), name="Conv2D_imageInput_1")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(32, (5,5), strides=(2,2), name="Conv2D_imageInput_2")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (5,5), strides=(2,2), name="Conv2D_imageInput_3")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_imageInput_4")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_imageInput_5")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Flatten(name="flattenedx")(x)
    x = Dense(100)(x)
    x = Dropout(rate)(x)
```

---

## Lane finding CNN

```python
# Preprocessed lane image input convnet
    y = laneInput
    y = Conv2D(24, (5,5), strides=(2,2), name="Conv2D_laneInput_1")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(32, (5,5), strides=(2,2), name="Conv2D_laneInput_2")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (5,5), strides=(2,2), name="Conv2D_laneInput_3")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_laneInput_4")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_laneInput_5")(y)
    y = LeakyReLU()(y)
    y = Flatten(name="flattenedy")(y)
    y = Dense(100)(y)
    y = Dropout(rate)(y)
```

---

## Concatenating the CNNs

```python
# Concatenated final convnet
    c = Concatenate(axis=1)([x, y])
    c = Dense(100, activation='relu')(c)
    c = Dense(50, activation='relu')(c)
```

---

## Outputs

```python
# Output layers
    steering_out = Dense(1, activation='linear', name='steering_out')(o)
    throttle_out = Dense(1, activation='linear', name='throttle_out')(o)
    model = Model(inputs=[imageInput, laneInput, behaviourInput], outputs=[steering_out, throttle_out]) 
    
    return model
```

---

## Training results
- 10k records made on the randomly generated track
- 12 epochs, 21m 45s on an RTX 2060
- The final validation loss was 0.003665.
<center><img src="/images/training.png" width="100%"></center>

---

## Preview

<center><video controls src="/videos/preview1.mp4" autoplay muted loop width="100%"></video></center>

{{%/section%}}