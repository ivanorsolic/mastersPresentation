+++
weight = 3
+++
{{% section %}}
## DonkeyCar

---

## What is Donkey?
Donkey is a high level self driving library written in Python.

---

## How was it used?

It was used as an interface between the RC car and the neural net that controls the car.

![](/images/donkey.png)


{{%/section%}}




<section>
<h2>Things Donkey solves:</h2>
{{% fragment %}} - ##### 📷Data preprocessing {{% /fragment %}}
{{% fragment %}} - ##### 🎮Controlling the RC car {{% /fragment %}}
{{% fragment %}} - ##### ✅Data collection/labeling {{% /fragment %}}
{{% fragment %}} - ##### 🏋️‍♂️Custom model training {{% /fragment %}}
{{% fragment %}} - ##### Question: RC Car only or Host PC + RC ❓{{% /fragment %}}
</section>

---


## Donkey RC calibration
---
<section data-background-video="/videos/steering_calibration.mp4" data-background-video-loop data-background-video-muted></section>
<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

#### Steering calibration
---
<section data-background-video="/videos/throttle_calibration.mp4" data-background-video-loop data-background-video-muted></section>
<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

#### Throttle calibration
---

<section data-background-video="/videos/gamepad_steering.mp4" data-background-video-loop data-background-video-muted></section>
<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

#### Gamepad steering
---

<section data-background-video="/videos/gamepad_throttle.mp4" data-background-video-loop data-background-video-muted></section>
<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

#### Gamepad Throttle

---

{{%section%}}
## Sanity check
![](/images/test_track2.png)

---

## First test track
![](/images/test_track.png)

---

## Fancier test track
![](/images/test_track_fancy.jpg)

---


## Collecting training data

<center><video controls src="/videos/collecting_data.mp4" autoplay muted loop width=100%></video></center>

---

## First basic autopilot

<center><video controls src="/videos/autopilot.mp4" autoplay muted  loop width=33%></video></center>
{{%/section%}}