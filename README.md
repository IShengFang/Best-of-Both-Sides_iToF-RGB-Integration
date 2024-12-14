# Best of Both Sides: Integration of Absolute and Relative Depth Sensing Modalities Based on iToF and RGB Cameras
[[paper]](https://people.cs.nycu.edu.tw/~walon/publications/fang2024icpr.pdf) [[supplementary materials]](https://people.cs.nycu.edu.tw/~walon/publications/fang2024icpr_supp.pdf) [[Springer]](http://dx.doi.org/10.1007/978-3-031-78444-6_30)

The official code for our ICPR2024 work, Best of Both Sides: Integration of Absolute and Relative Depth Sensing Modalities Based on iToF and RGB Cameras.

![](./teaser.gif)

## Requirements
We highly recommend using the [Conda](https://docs.anaconda.com/miniconda/) to build the environment. 

You can build and activate the environment by following commands. 
```
conda env create -f environment.yml 
conda activate iToF-RGB
```

We provide three environment files for different hardware settings.
- For GPU with CUDA 10.1 (e.g. RTX 2080Ti), use `environment.yml`.
- For GPU with CUDA 11.6+ (e.g. RTX 3090), use `environment_cu11.yml`
- For CPU only, use `environment_cpu.yml`

## Dataset 
You can download ToF-FlyingThings3D from [here](https://drive.google.com/drive/folders/1XASaOfcp3TzQJ0A2fMaXex-0eihha0vg?usp=sharing), provided by [Qiu et al.](https://github.com/sylqiu/tof_rgbd_processing).

After downloading the dataset, you need to preprocess the dataset by running the following command. 
```
mv train_list.txt <dataset_root>/ToF-FlyThings3D/
mv test_list.txt <dataset_root>/ToF-FlyThings3D/
```
### Validation split

Run the following command to split the training set into training and validation sets. 
```
python split_train_val.py --dataset_root <dataset_root>
```

## Training
You can train the model by running the following command. 

```
python train.py --config_path <configuration yaml> --generate_exp_name
```
The `--generate_exp_name` flag will generate the experiment name based on the configuration file.

The `--train_cam_model` flag is used to specify the model for which camera to train.

### Configuration
You can find the configuration yaml files in the `configs` directory.
These yaml files contain the hyperparameters for training.
For pseudo camera calibration, you can use the configuration files starting with `pseudo`.
For device camera calibration, you can use the configuration files starting with `device`.

We provide four different modes for training model.
`RGBD` mode is for training each model by inputting RGB image and iToF depth and outputting the depth of each camera view.
`StereoRGB` mode is for training the model with RGB stereo images and output the depths of both camera views.
`SplitStereo` mode is for training each model with RGB stereo images and output the depth of each camera view.
`MonoRGB` mode is for training the model with monocular RGB image and output the depth of the camera view.

### Camera parameters
The camera calibration parameters are stored in the `cam_param` directory for device camera configuration and `cam_param_psuedo` camera configuration.

## Evaluation
You can evaluate the model by running the following command. 

```
python test.py --config_path <configuration yaml> --test_cam <test model for which camera (W, UW, or both)> --uw_model_path <path to the UW model> --w_model_path <path to the W model>
```

Notice that the `StereoRGB` models need to be evaluated with the `--stereo_weight_path` as follows. 
```
python test.py --config_path <configuration yaml> --stereo_weight_path <path to the StereoRGB model>
```

## Citation
```Bibtex
@inproceedings{fang2024itofrgb,
 title = {Best of Both Sides: Integration of Absolute and Relative Depth Sensing Modalities Based on iToF and RGB Cameras},
 author = {I-Sheng Fang and Wei-Chen Chiu and Yong-Sheng Chen},
 booktitle = {ICPR},
 year = {2024}
} 
```

## Disclaimer
This repository is for research purposes only.
