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

## Config

Before you run codes below, please correct information in config file ```config.py```

## Calibration

Preview and save calibration pictures: 

```
python calibration.py
```

## Rectification

Calculate and save the rectification matrixs:

```
python rectification.py
```

## Depth Estimation

```
python stereo_zoo.py
```

In file ```stereo_zoo.py```, class ```StereoSegmentationPredictor``` is used to get target's mask. class ```StereoObjectDetector``` is used to detect the target. class ```StereoFeatureMatcher``` is used to perform image matching. class ```StereoDepthEstimator``` is used to wrap them together.

## TODO

- ~~Use bbox to reduce the range of matches(both images)~~
- ~~Use edges to determine feature points~~ (Quit, bad performance)
- ~~add ```config``` file~~
- ~~rewrite codes as class~~
- check the correctness of calculate coords of x and y in cropped images (use standard cube or others, calculate its length)