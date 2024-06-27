# Stereo Depth Estimation

## Quickstart

Clone repo:

```
git clone --recurse-submodules https://github.com/cuteStarrr/StereoVision.git
cd StereoVision
```

Install requirements:

```
conda create -n <your_env_name> python=3.11
conda activate <your_env_name>
pip install -r requirements.txt
```

## Calibration

Preview and save calibration pictures: 

```
python calibration.py
```

## Depth Estimation

```
python distance_measurement.py
```

In file ```distance_measurement.py```, function ```stereo_calibration``` is used to get the camera's internal and external parameter matrix. function ```disparity_calculation``` is used to calculate the depth of video in real time.

function ```disparity_calculation``` uses ```efficientSAM``` to get the target segmentation, thus acquiring more accurate depth estimation.

## TODO

- Use bbox to reduce the range of matches(both images)
- Use edges to determine feature points