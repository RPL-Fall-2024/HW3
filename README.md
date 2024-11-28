# RPL HW3: Reinforcement learning of robot policies


## ️ Installation

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 

Activate the same environment as hw2
```console
> conda activate rpl-hw2
> pip install gym==0.26.2
> pip install gymnasium==0.29.1
> pip install --upgrade stable_baselines3[extra]==2.3.2
```
or build the new environment and reinstall the packages
```
pip install -r requirements.txt
```


## Training
To train the model, run the command:
```
bash scripts/train.sh 
```
