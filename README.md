![](figures/model.png)

<font size=4> We propose a novel method, SANPRO, for.  </font> <br><br>


# SANPRO

The official implementation for "**SANPRO**".

**Table of Contents**

* [Datasets](#Datasets)
* [Installation](#Installation)
* [Usage](#Usage)
* [Tutorial](#Tutorial)
* [Citation](#Citation)

## Datasets


We provide an easy access to the used datasets in the [synapse](https://www.synapse.org/#!Synapse:syn52559388/files/).


## Installation

To reproduce **SANPRO**, we suggest first create a conda environment by:

~~~shell
conda create -n SANPRO python=3.8
conda activate SANPRO
~~~

and then run the following code to install the required package:

~~~shell
pip install -r requirements.txt
~~~

and then install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) according to the CUDA version, take torch-1.13.1+cu117 (Ubuntu 20.04.4 LTS) as an example:

~~~shell
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
~~~

## Usage

### data preprocessing


In order to run **SANPRO**, we need to first create anndata from the raw data.

The h5ad file should have cells as obs and peaks as var. There should be at least three columns in `var`:  `chr`, `start`, `end` that indicate the genomic region of each peak. The h5ad file should also contain two columns in the `obs`: `Batch` and `CellType` （reference data）, where `Batch` is used to distinguish between reference and query data, and `CellType` indicates the true label of the cell.


### Stage 1: embeddings extraction

The processed data are used as input to *scbasset* and a reference genome is provided to extract the embedding incorporating sequence information: 

~~~shell
# Stage 1: embeddings extraction

python train_demo.py
~~~


### Stage 2: batch effect removal

~~~shell
# Stage 2: batch effect removal

python remove_batch_demo.py 
~~~


## Tutorial

### Tutorial 1: Cell annotations within samples (LargeIntestineB_LargeIntestineA)
1. Install the required environment according to [Installation](#Installation).
2. Create a `data` folder in the same directory as the 'SANGO' folder and download datasets from [LargeIntestineA_LargeIntestineB.h5ad](https://www.synapse.org/#!Synapse:syn52559388/files/).
3. Create a folder `genome` in the ./SANGO/CACNN/ directory and download [mm9.fa.h5](https://www.synapse.org/#!Synapse:syn52559388/files/).
4. For more detailed information, run the tutorial [LargeIntestineB_LargeIntestineA.ipynb](LargeIntestineB_LargeIntestineA.ipynb) for how to do data preprocessing and training.


### Tutorial 2: Cell annotations on datasets cross platforms (MosP1_Cerebellum)
1. Install the required environment according to [Installation](#Installation).
2. Create a `data` folder in the same directory as the 'SANGO' folder and download datasets from [MosP1_Cerebellum.h5ad](https://www.synapse.org/#!Synapse:syn52559388/files/).
3. Create a folder `genome` in the ./SANGO/CACNN/ directory and download [mm10.fa.h5](https://www.synapse.org/#!Synapse:syn52559388/files/).
4. For more detailed information, run the tutorial [MosP1_Cerebellum.ipynb](MosP1_Cerebellum.ipynb) for how to do data preprocessing and training.




### Tutorial 3: Cell annotations on datasets cross tissues (BoneMarrowB_Liver)
1. Install the required environment according to [Installation](#Installation).
2. Create a `data` folder in the same directory as the 'SANGO' folder and download datasets from [BoneMarrowB_Liver.h5ad](https://www.synapse.org/#!Synapse:syn52559388/files/).
3. Create a folder `genome` in the ./SANGO/CACNN/ directory and download [mm9.fa.h5](https://www.synapse.org/#!Synapse:syn52559388/files/).
4. For more detailed information, run the tutorial [BoneMarrowB_Liver.ipynb](BoneMarrowB_Liver.ipynb) for how to do data preprocessing and training.


### Tutorial 4: Multi-level cell type annotation and unknown cell type identification
1. Install the required environment according to [Installation](#Installation).
2. Create a `data` folder in the same directory as the 'SANGO' folder and download datasets from [BCC_TIL_atlas.h5ad, BCC_samples.zip, HHLA_atlas.h5ad](https://www.synapse.org/#!Synapse:syn52559388/files/).
3. Create a `genome` folder in the same directory as the 'SANGO' folder and download [GRCh38.primary_assembly.genome.fa.h5](https://www.synapse.org/#!Synapse:syn52559388/files/).
4. For more detailed information, run the tutorial [tumor_example.ipynb](tumor_example.ipynb) for how to do data preprocessing and training.


## Citation

If you find our codes useful, please consider citing our work:

~~~bibtex

~~~
