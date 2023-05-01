# DeepSE-WF: Unified Security Estimation for Website Fingerprinting Defenses

This is the code for evaluating Website Fingerprinting (WF) defenses, associated with the paper "DeepSE-WF: Unified Security Estimation for Website Fingerprinting Defenses."

[![(A. Veicht, 2022)](http://img.shields.io/badge/paper-arxiv.2206.08672-B31B1B.svg)](https://arxiv.org/abs/2203.04428)

## Citation
If you use this code or want to build on this research, please cite our paper:

```BibTeX
@article{deepse-wf,
  title={DeepSE-WF: Unified Security Estimation for Website Fingerprinting Defenses},
  author={Veicht, Alex and Renggli, Cedric and Barradas, Diogo},
  journal={Proceedings on Privacy Enhancing Technologies},
  year={2023},
  volume={2023},
  number={2},
  address={Lausanne, Switzerland}
}
```

## Background
Background:
Website fingerprinting (WF) attacks have been a growing concern in the field of network security. These attacks, carried out by an eavesdropper on a network, can accurately identify the websites visited by a user by analyzing their traffic patterns, even when the user is accessing the internet through encrypted channels such as Tor or VPNs. This makes WF attacks a serious threat to user privacy and anonymity online.

To counter this threat, several defenses have been proposed in recent years, including randomized packet padding, traffic morphing, and obfuscation techniques. However, the effectiveness of these defenses is often difficult to assess, as attackers can adapt their strategies to bypass them.

To evaluate the security of these defenses, previous works have proposed feature-dependent theoretical frameworks that estimate the Bayes error or mutual information leaked by manually-crafted features. However, as WF attacks increasingly rely on deep learning and latent feature spaces, these frameworks can no longer provide accurate security estimations.

To address this issue, this work proposes DeepSE-WF, a novel WF security estimation framework that leverages specialized kNN-based estimators to produce Bayes error and mutual information estimates from learned latent feature spaces. This approach bridges the gap between current WF attacks and security estimation methods and produces tighter security estimates than previous frameworks.

## Setup

The code works for [Python 3.8.5](https://www.python.org/downloads/release/python-385/). All examples assume a unix shell. First install the requirements using [pip](https://pip.pypa.io/en/stable/installation/)

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

or using [conda](https://docs.conda.io/en/latest/)

```
conda env create -f environment.yml
conda activate deepse
```

If none of the above work, you may use [docker](https://www.docker.com) with the provided [Dockerfile](Dockerfile). The following commands will build the image and execute into the container. From there, you can run the commands as described below. Note that the performance of the docker container may be  worse than the native installation. All data will be deleted when the container is stopped.

```
docker build -t deepse .
docker run -it deepse
```

If you are using docker and want to recreat the plots, run the following command:

```
docker run -p 8888:8888 -v $(pwd)/plots:/home/jovyan/ jupyter/scipy-notebook
```

This will start a jupyter notebook server. After the server has started, it will print a link to the console. Open this link in your browser. There you can find the notebook to recreate the plots.

# Estimating the Security of Website Fingerprinting Defenses

This section describes how to estimate the security of website fingerprinting (WF) defenses using the DeepSE-WF framework.
## Dataset Format

We consider the dataset collected to consist of files where each trace is stored in the form `$W-$T`, where `$W` is the website index and `$T` is the trace index (both starting from 0). For example, "1-3" is the fourth page load of the second website.

Each of these files contains, per row:
    
    t_i<tab>s_i

with t_i and s_i indicating respectively time and size of the i-th packet.
The sign of s_i indicates the packet's direction (positive means outgoing).
Note: because this dataset represents Tor traffic, where packets' sizes are
fixed, s_i will effectively only indicate the direction, taking value in
{-1, +1}.

## Preparing the Dataset
Either generate your own data with your defense of download the AWF dataset following the instructions in [Prepare the AWF Dataset](#prepare-the-awf-dataset) and simulate your defense on the dataset.

### Preparing the Data for DeepSE-WF

The [create_dataset.py](preprocessing/create_dataset.py) script takes all the trace files, brings them into the correct format and saves them into a numpy matrix. The data is stored in a ```.npz``` file which contains the arrays ```traces``` and ```labels```. For example:

```
python preprocessing/create_dataset.py \
  --in_path <path-to-folder-containing-processed-traces> \
  --out_path <output-file> \
  --n_websites <number-of-websites-to-use> \
  --n_traces <number-of-traces-to-use-per-website>
```

## Measuring the Security
In order to estimate the security for a specific defense, simply run main.py. The results will stored in the file specified by --log_file. For example, we can estimate the security for the dataset from above as follows:
  
  ```
  python main.py \
  --dataset_path <path-to-dataset.npz> \
  --n_traces <number-of-traces-per-website> \
  --log_file <output-file>
  ```

  This will run 5-fold cross validation and report the Bayes Error Rate and Mutual Information estimation in the log.txt file. If you have a GPU, you can use it by adding --device cuda and if you have multiple GPU's on your machine, you can select one using --gpu_id id.

# Example Usage

In order to reproduce the results, you can either download the preprocessed traces or follow the instructions in [Prepare the AWFDataset](#prepare-the-awf-dataset) to generate the dataset yourself.

## Downloading Preprocessed Dataset
The preprocessed awf dataset (100 websites and 4500 traces each) is available [here](https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z). It can also be downloaded using the following commands (**This will take about 22 GB of disk space**):

```
mkdir -p data/dataset/awf
wget -O data/dataset/awf/NoDef.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FAWF-100-4500\&files\=NoDef.npz
wget -O data/dataset/awf/wtfpad.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FAWF-100-4500\&files\=wtfpad.npz
wget -O data/dataset/awf/Front_T1.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FAWF-100-4500\&files\=Front_T1.npz
wget -O data/dataset/awf/Front_T2.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FAWF-100-4500\&files\=Front_T2.npz
wget -O data/dataset/awf/cs_buflo.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FAWF-100-4500\&files\=cs_buflo.npz
wget -O data/dataset/awf/tamaraw.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FAWF-100-4500\&files\=tamaraw.npz
```

This will download the preprocessed traces in the `data/datasets/awf` folder.

The DS19 dataset is available at the same link and can be downloaded using the following commands (**This will take about 1 GB of disk space**):

```
mkdir -p data/dataset/ds19
wget -O data/dataset/ds19/NoDef.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FDS19-100-100\&files\=NoDef.npz
wget -O data/dataset/ds19/wtfpad.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FDS19-100-100\&files\=wtfpad.npz
wget -O data/dataset/ds19/Front_T1.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FDS19-100-100\&files\=Front_T1.npz
wget -O data/dataset/ds19/Front_T2.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FDS19-100-100\&files\=Front_T2.npz
wget -O data/dataset/ds19/cs_buflo.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FDS19-100-100\&files\=cs_buflo.npz
wget -O data/dataset/ds19/tamaraw.npz https://polybox.ethz.ch/index.php/s/2hyigdcNv33y33z/download\?path\=%2FDS19-100-100\&files\=tamaraw.npz
```

Once you have downloaded the dataset, you can directly run the experiments as described in [Measuring Security](#measuring-security).

## Prepare the AWF Dataset
The dataset by [Rimmer et al.](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-1_Rimmer_paper.pdf) can be downloaded from [here](https://distrinet.cs.kuleuven.be/software/tor-wf-dl/files/closed_world_csvs.tar.gz) or using the following command (**This will take about 90 GB of disk space**):

```
wget https://distrinet.cs.kuleuven.be/software/tor-wf-dl/files/closed_world_csvs.tar.gz
```

Then, extract the collection of .tar.gz files using the following command (**This will take another 90 GB of disk space**):

```
tar -xvzf closed_world_csvs.tar.gz
```

Currently, ```tor_run_v1_000.tar.gz``` seems to be corrupted. You can remove it using the following command:

```
rm closed_world/tor_run_v1_000.tar.gz
```

In order to free up some space, you can remove the ```closed_world_csvs.tar.gz``` file using the following command:

```
rm closed_world_csvs.tar.gz
```

The dataset is provided as a collection of .tar.gz files, each containing a set of websites and traces. The dataset can be extraced and cleaned using  [extract_awf_tar.py](preprocessing/extract_awf_tar.py):

```
python preprocessing/extract_awf_tar.py \
  --in_path <path-to-folder-containing-tar-files> \
  --out_path <output-folder>
```

e.g.
  
```
python preprocessing/extract_awf_tar.py \
  --in_path /Downloads/closed_world/ \
  --out_path data/awf
```

This will read all `.tar.gz` files in the input directory and extract them to the output directory. The output directory will contain a subdirectory for each website, containing the traces for that website.

### Preprocessing the Traces
In order to create a clean dataset, we need to preprocess the traces. This is done using [create_nodef.py](preprocessing/create_nodef.py):

```
python preprocessing/create_nodef.py \
  --in_path <path-to-folder-containing-extracted-websites> \
  --out_path <output-folder> \
  --n_websites <number-of-websites-to-use> \
  --n_traces <number-of-traces-to-use-per-website>
```

e.g.

```
python preprocessing/create_nodef.py \
  --in_path data/awf \
  --out_path data/traces/NoDef_awf \
  --n_websites 100 \
  --n_traces 100
```

This will first count the available traces and websites, and then create a new dataset with the specified number of websites and traces if enough websites/traces are available. The output directory will contain all traces in the form `$W-$T`, where `$W` is the website index and `$T` is the trace index (both starting from 0).

## Prepare the DS19 Dataset
The dataset by [Wang et al.](https://www.cs.sfu.ca/~taowang/wf/Go-FRONT.pdf) can be downloaded from [here](https://www.cs.sfu.ca/~taowang/wf/index.html) or using the following command:

```
wget https://www.cs.sfu.ca/\~taowang/wf/20000.zip
```

In order to prepare the dataset, unzip the data:

```
unzip 20000.zip
rm 20000.zip
```
  
Then, move the traces to a new folder `data/traces/DS19` and remove the old folder:
  
```
mkdir -p data/traces/DS19
mv 20000/*-*.cell data/traces/DS19
rm -rf 20000
```

Finally remove the ```.cell``` extension from all files:

```
for f in data/traces/DS19/*; do mv "$f" "${f%.cell}"; done
```

## Defending the Dataset

In order to defend the dataset, you can run the [simulate_defenses.sh](defenses/simulate_defenses.sh) script with the path to the undefended traces as input

```
bash simulate_defenses.sh <path-to-undefended-traces>
```

e.g.

```
bash simulate_defenses.sh ../data/traces/NoDef_awf
```

Make sure that you are in the defense folder (```pwd``` should end with  ```DeepSE-WF/defenses```). This will run all defenses and store the results in the `data/defended` folder.


## Preparing the Data for DeepSE-WF

Preparation of the data for DeepSE-WF is done using [create_dataset.py](preprocessing/create_dataset.py):

```
python preprocessing/create_dataset.py \
  --in_path data/traces/NoDef_awf \
  --out_path data/dataset/awf/NoDef.npz \
  --n_websites 100 \
  --n_traces 100
```

loads 100 website and 100 traces per website from from ```data/traces/NoDef``` and stores them into ```data/dataset/NoDef.npz```.

## Measuring Security
In order to estimate the security for a specific defense, simply run [main.py](main.py):

```
python main.py --data_path data/dataset/awf/NoDef.npz \
  --n_traces 100 \
  --log_file log.txt
```

This will run 5-fold cross validation and report the Bayes Error Rate and Mutual Information estimation in the ``log.txt`` file.

## Creating the Plots
All the plots can be recreated using the [plots/create_plots.ipynb](plots/create_plots.ipynb) notebook. The plots will be stored in the ``plots/outputs`` directory. The results from the paper are stored in the ``plots/values`` directory which are copied values from the log files.

In order to recreate the plots, you may adapte the main function to store the relevant information to a ```.csv``` with the corresponding columns from the tables in ```plots/values``` or update the current values manually. Then you can use the notebook to recreate the plots.

## Acknowledgements
We use the codebase by [Cherubin et al](https://github.com/gchers/wfes) for the [wfes](https://www.petsymposium.org/2017/papers/issue4/paper50-2017-4-source.pdf) estimations as well as the [tamaraw](https://dl.acm.org/doi/10.1145/2660267.2660362) and [cs-buflo](https://dl.acm.org/doi/10.1145/2382196.2382260) defense implementations. We also use the codebase by [Gong et al.](https://github.com/websitefingerprinting/websitefingerprinting) for the [Front](https://www.usenix.org/conference/usenixsecurity20/presentation/gong) defense implementation as well as the codebase by [Rahman et al.](https://github.com/notem/reWeFDE) for the Mutual information estimation by [wefde](https://dl.acm.org/doi/10.1145/3243734.3243832).

The [df](https://dl.acm.org/doi/10.1145/3243734.3243768) implementation is based on the [this](https://github.com/deep-fingerprinting/df) repository, the [awf](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-1_Rimmer_paper.pdf) implementation is based on the [this](https://github.com/DistriNet/DLWF) repository and the [tf](https://dl.acm.org/doi/10.1145/3319535.3354217) implementation is based on the [this](https://github.com/triplet-fingerprinting/tf) repository. Finally, the [var_cnn](https://petsymposium.org/2019/files/papers/issue4/popets-2019-0070.pdf) implementation is based on the [this](https://github.com/sanjit-bhat/Var-CNN) repository.

### Commit Hash
The commit hash of the code used for the paper is ```c2a915f4117531ec7ae80092b4ccbefa51591479```.
