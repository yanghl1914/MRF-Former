# SPP ChangeFormer

## How to use this code
Adjust the relevant parameters in config.yml before running training
### for single device
``` python
pthon3 main.py
```

###for multi devices
Call parameter adjust equipment in gpus.sh and 
``` python
sh gpus.sh
```

## Files information
### config.yml
Adjustable parameter file. Note: 
to facilitate validation, resize is done for data of different sizes, controlled by the parameter 'datasets.resize'.

### main.py
Main implementation documents

### gpus.sh
Multi devices call start script

### models
All models files included in src/models.
SPP models are in ChangeFormerV6.py(old) and new_spp.py(new)
Original ChangeFormer modle is in Other_ChangeFormer.py

### datasets
datasets please set in /datasets, and modify the parameter 'dataset.root_dir' in config.yml.

### utils in /src
Data pre-processing and dataloader generation functions are included in 'loader.py'  
The loss functions are included in 'losser.py'  
Precision verification functions are included in 'metrics.py'  
Cosine annealing optimizer is included in 'optimizer.py'  
other utils such as logger are included in 'utils.py'
