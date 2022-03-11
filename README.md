# DeepSE-WF: Unifier Security Estimation for Website Fingerprinting Defenses

This is for evaluating Website Fingerprinting (WF) defenses, associated with the paper "DeepSE-WF: Unified Security Estimation for Website Fingerprinting Defenses."
[(A. Veicht, 2022)](https://arxiv.org/abs/2203.04428).

## Installation

The code works for Python 3.8.5. First install the requirements using pip

```
pip install -r requirements.txt
```

or using conda

```
conda create -n deepse python=3.8.5
conda activate deepse
conda install --file requirements.txt
```

## The dataset

We consider the dataset collected to consist of files where each trace is stored in the form `$W-$T`, where `$W` is the website index and `$T` is the trace index (both starting from 0). For example, "1-3" is the fourth page load of the second website.

Each of these files contains, per row:
    
    t_i<tab>s_i

with t_i and s_i indicating respectively time and size of the i-th packet.
The sign of s_i indicates the packet's direction (positive means outgoing).
Note: because this dataset represents Tor traffic, where packets' sizes are
fixed, s_i will effectively only indicate the direction, taking value in
{-1, +1}.


## Defending the dataset

Link to defense simulations or include them here.


## Preparing the data

The [createDataset.py](utils/createDataset.py) script takes all the trace files, brings them into the correct format and saves them into a numpy matrix. The traces are stored in traces.npy and the corresponding labels are stored in labels.npy. For example, 

```
python utils/createDataset.py --trace_path .../dataset/NoDef \
  --save_path .../dataset/NoDef_100_60 \
  --num_classes 100 \
  --n_traces 60
```

loads 100 website and 60 traces per website from from ```.../dataset/NoDef``` and stores them into ```.../dataset/NoDef_100_60/traces.npy``` and ```.../dataset/NoDef_100_60/labels.npy```

## Measuring security
In order to estimate the security for a specific defense, simply run [main.py](main.py). The results will stored in the file specified by ``--log_file``. For example, we can estimate the security for the dataset from above as follows:

```
python main.py --data_path .../dataset/NoDef_100_60 \
  --n_traces 60 \
  --log_file log.txt
```

This will run 5-fold cross validation and report the Bayes Error Rate and Mutual Information estimation in the ``log.txt`` file.

