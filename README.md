![](model.jpg)

<font size=4> We present SANPRO, a novel method for the analysis of single-cell and spatial proteomics data.


# SANPRO

The official implementation for "**SANPRO**".

**Table of Contents**

* [Datasets](#Datasets)
* [Installation](#Installation)
* [Usage](#Usage)
* [Citation](#Citation)

## Datasets


We provide some data demos in the fold named "data_demo". The complete dataset can be downloaded via the download link provided in the paper.


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

## Citation

If you find our codes useful, please consider citing our work:

~~~bibtex


@article{zengSANPRO,
  title={SANPRO: A Cross-scale Framework for Integrating mRNA Sequences and Protein Abundance in Decoding Single-cell and Spatial Proteomics},
  author={Yuansong Zeng, Jianing Chen, Wenbing Li, Hongcheng Chen, Hongyu Zhang, Huiying Zhao, Zheng Wang, Yuedong Yang},
  journal={},
  year={2025},
}
~~~
