# Modeling Multivariate Biosignals With Graph Neural Networks and Structured State Space Models

Siyi Tang, Jared A. Dunnmon, Liangqiong Qu, Khaled K. Saab, Tina Baykaner, Christopher Lee-Messer, Daniel L. Rubin. *arXiv*. https://arxiv.org/abs/2211.11176

---
## Setup
This codebase requries python ≥ 3.9, pytorch ≥ 1.12.0, and pyg installed. 

Please refer to [PyTorch installation](https://pytorch.org/) and [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 

Other dependencies are included in `requirements.txt` and can be installed via `pip install -r requirements.txt`

---
## Datasets
### DOD-H
The DOD-H dataset is publicly available and can be downloaded from this [repo](https://github.com/Dreem-Organization/dreem-learning-open).

Currently, downloaded DOD-H dataset is saved in `/HDD/dodh`

### Our PSG data
We used PSG dataset in `/nas/SNUBH-PSG_signal_extract/train_data`
### Preprocessing our PSG data

**Please note that this process takes lots of times and space, so check if your save-dir have enough space before proceeding.**
 
To preprocess/save our PSG data, specify `<dir-to-PSG-data>`, `<your-save-dir>`, then run:
```
python ./signal_process/psgsync.py --data_dir <dir-to-PSG-data> --output_dir <your-save-dir>
```

---
## Model Training
`scripts` folder shows examples to train GraphS4mer on the three datasets. If you have a GPU with smaller memory, you can decrease the batch size and set `accumulate_grad_batches` to a value > 1. 

### Model training on DOD-H dataset
To train GraphS4mer on the DOD-H dataset, specify `<dir-to-dodh-data>` and `<your-save-dir>` in `scripts/run_dodh.sh`, then run:
```
bash ./scripts/run_dodh.sh
```
### Model training on Our dataset
To train GraphS4mer on Our dataset, 
1. Run this first ***(Only if you want to create all data1~3 filemarkers)*** to create file_markers for the training.
```
python create_filemarker.py
```
2. Specify filemarker directory `DODH_FILEMARKER_DIR` in `datamodule_dreem.py`
3. Specify `logger` for the name of the tensorboard logger in `train.py` **(which is in main)**
4. Check `constants.py` to see `OURS_CHANNELS` list has appropriate sensors for the training

   *(It is set to 6EEG+2EOG+1EKG+1EMG)*

5. specify these follwings in `scripts/run_ours.sh`:

`RAW_DATA_DIR` : directory to our processed PSG data

`SAVE_DIR` : directory to model checkpoints

`sampling_freq` : sampling frequency for the resampling **(all channels will be resampled to this frequency)**

`num_nodes` : number of channels for the training **(It has to match with the length of `OURS_CHANNELS` in `constants.py`)**

`resolution` : Time steps for each dynamic graph to be made.

   *(e.g., 250Hz 30s -> 7500 steps, if resolution : 2500, it means 3 dynamic graphs are made, each graph is responsible for 10s)*
      
`gpus` : For multi-gpu training, this number should be higher than 1

6. run this for the training:

```
bash ./scripts/run_ours.sh
```

---
## Utils


---
## Reference
If you use this codebase, or otherwise find our work valuable, please cite:
```
@misc{tang2023modeling,
      title={Modeling Multivariate Biosignals With Graph Neural Networks and Structured State Space Models}, 
      author={Siyi Tang and Jared A. Dunnmon and Liangqiong Qu and Khaled K. Saab and Tina Baykaner and Christopher Lee-Messer and Daniel L. Rubin},
      year={2023},
      eprint={2211.11176},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
