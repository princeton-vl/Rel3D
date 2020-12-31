<img src="rel3d.gif" align="right" width="30%"/>

[**Rel3D: A Minimally Contrastive Benchmark for Grounding Spatial Relations in 3D**](https://arxiv.org/pdf/2012.01634.pdf)
[Ankit Goyal](http://imankgoyal.github.io), [Kaiyu Yang](https://www.cs.princeton.edu/~kaiyuy/), [Dawei Yang](http://www-personal.umich.edu/~ydawei/), [Jia Deng](https://www.cs.princeton.edu/~jiadeng/) <br/>
***Neural Information Processing Systems (NeuRIPS), 2020 (Spotlight)***

## Getting Started

First clone the repository. We would refer to the directory containing the code as `Rel3D`.

```
git clone git@github.com:princeton-vl/Rel3D.git
```

#### Requirements
The code is tested on Linux OS with Python version **3.6.9**, CUDA version **10.2**.

#### Install Libraries
We recommend you to first install [Anaconda](https://anaconda.org/) and create a virtual environment.
```
conda create --name rel3d python=3.6
```

Activate the virtual environment and install the libraries. Make sure you are in `Rel3D`.
```
conda activate rel3d
pip install -r requirements.txt
conda install sed
```

#### Download Datasets and Pre-trained Models
Make sure you are in `Rel3D`. `download.sh` script can be used for downloading all the data and the pretrained models. It also places them at the correct locations. First, use the following command to provide execute permission to the `download.sh` script. 
```
chmod +x download.sh
```

To download the data sufficient for running all experiments in Table 1, execute the following command. It will download only the primary split of the data (~2GB) that is used in Table 1.
```
./download.sh data_min
```

To download the data for running all experiments (i.e. Table 1 and Fig. 5), execute the following command. It will download all different splits of the data (~8GB) which are required for running the `Contrastive vs Non-Contrastive` experiments with varying dataset sizes. It will also download the primary split.
```
./download.sh data
```

To download the pretrained models, execute the following command.
```
./download.sh pretrained_model
```

To download the raw data, execute the following command. It places the data in the `data/20200223`. For each sample there is a `.pkl`, `.png` and `.tiff` file. The `.png` and `.tiff` files store rgb and depth respectively at 720X1280 resolution. Information about object masks, bounding box and surface normal are stored in the `.pkl` file. Note that the `./download.sh data` downloads the rgb and depth images in a compressed format, which is sufficient to reproduce all the experiments. The raw data is much larger and might not be necessary for most use cases.

*WARNING: You also need to execute `./download.sh data` or `./download data_min` to download the `<split>.json` files (described later). All information like spatial relation and object category should be parsed using the `<split>.json` files and not from the file names.*

```
./download.sh data_raw
```

**If you get error while executing the above command, you can manually download the data using the [link](https://drive.google.com/uc?id=1MSMwnX0znCfgEisj7zJ4ohFWJDrsxeme). After downloading the zip file, you need to extract it and place the extracted `20200223` folder inside the `data` folder.**

## Data Organization
All data to run the models is in the `Rel3D/data` folder. 

The raw images are stored in the `Rel3D/data/20200223` folder (in case you downloaded them).

There are 7 splits for the complete dataset. If you used `./download.sh data_min`, you would have only the primary split. If you used `./download.sh data`, you would have all the 7 splits. 

Each split is named as `<c/nc>_<per_train>_<c/nc>_<per_valid>`. Here `c` stands for contrastive and `nc` stands for non-contrastive. For example, the `<nc>_<0.4>_<nc>_<0.1>` split means that the training and validation samples are non-contrastive, and `40%` of the complete dataset is used for training while `10%` is used for validation. All experiments in `Table 1` are conducted using the `c_0.9_c_0.1` split. The other 6 splits are used to conduct the `Contrastive vs Non-Contrastive` experiments shown in `Figure 5` of the paper. The testing data is the same for all splits. 

For each split, there are 10 files. The `<split>.json` stores information about each split in the `json` format. Each sample is represented as a dictionary, with different keys storing various information like rgb image path (`rgb`), depth image path (`depth`), information about the camera used for rendering the image (`camera_info`), image dimensions (`width`, `height`), subject (`subject`), object (`object`), spatial relation (`predicate`), whether the spatial relation holds (`label`), and the simple 3D features we extracted for experiments in Section 5 (`transform_vector`). 

We also have `<split>_<train/test/valid/stats>_<crop_or_not>.h5` files for each split. They contain the pre-processed rgb and depth images in a compressed format. This allows us to load the entire dataset in memory, which speeds up training. If the `*.h5` files are not present in the  `Rel3D/data`, they are generated on-the-fly using the raw images, as described [here](https://github.com/princeton-vl/Rel3D/blob/master/dataloader.py#L198-L321).

You can visualize the samples with just the `*.h5` files and even without downloading the raw data. For this, use the following command:
```
python dataloader.py
```
This will run the `__main__` [function](https://github.com/princeton-vl/Rel3D/blob/master/dataloader.py#L654-L669) inside  the `dataloader.py` and save samples in the `Rel3D` directory. You can edit the arguments inside the `__main__` function depending on your need. [This](https://github.com/princeton-vl/Rel3D/blob/master/dataloader.py#L440-L463) part of the dataloader code generates the visualizations.

## Code Organization
- `Rel3D/models`: PyTorch model code for various models in PyTorch.
- `Rel3D/configs`: Configuration files for various models.
- `Rel3D/main.py`: Training and testing any models.
- `Rel3D/configs.py`: Hyperparameters for different models and dataloader.
- `Rel3D/dataloader.py`: Code for creating a PyTorch dataloader for our dataset.
- `Rel3D/utils.py`: Code for various utility functions.
 
## Running Experiments

#### Training and Testing
To train, validate, and test any model, we use the `main.py` script. The format for running this script is as follows. 
```
python main.py --exp-config <path to the config>
```

`exp-config` contains all information about the experiment. It contains the training hyper-parameters, model hyper-parameters as well as the dataloader hyper-parameters. The default value for each hyper-parameter is defined in `configs.py`. These default values are overwritten by the values in the `exp-config`. We provide `exp-config` for each model in `Table 1`. These configs can be found in the `Rel3D/configs` folder. As a concrete example, to execute the experiment for the DRNet model, use the command `python main.py --exp-config ./configs/drnet.yaml`. To execute a new experiment with a different hyperparameter, one needs to create a configuration file. 

The `python main.py --exp-config <path to the config>` command stores all the training logs in the `Rel3D/runs/EXP_ID` folder. The `EXP_ID` is specified in the `exp-config`. The best performing model on the validation set is saved as `Rel3D/runs/EXP_ID/model_best.pth`. The performance of this model is used for reporting results.

#### Evaluate a pretained model
We provide pretrained models. They can be downloaded using the `./download pretrianed_models` command and are stored in the `Rel3D/pretrained_model` folder. To test a pretrained model use the following command. The `<model_name>` has to be one either `2d`, `drnet`, `mlp_aligned`, `mlp_raw`, `pprfcn`, `vipcnn` or `vtranse`. Note that since we retrained the models, there are small differences (+- 0.5\%) in performance from the reported numbers in the paper. 
```
python main.py --entry test --exp-config configs/<model_name>.yaml --model-path pretrained_models/<model_name>.yaml
```

To render images from the 3D data, please use the [Rel3D_Render](https://github.com/princeton-vl/Rel3D_Render) repository. It also contains information about extracting 3D features which we used in our MLP baseline. (Table 1, Column8-9) 

If you find our research useful, consider citing it:
```
@article{goyal2020rel3d,
  title={Rel3D: A Minimally Contrastive Benchmark for Grounding Spatial Relations in 3D},
  author={Goyal, Ankit and Yang, Kaiyu and Yang, Dawei and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
