# SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection

If you use this code for your own research, you must reference our conference paper:

```
SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection 
Ricardo Pizarro, Roberto Valle, José M. Buenaposada, Luis M. Bergasa, Luis Baumela.
Proc. European Conference on Computer Vision, ECCV 2026.
```

#### Requisites
**CUDA 11.8** and **cuDNN 9** support — CUDA 11.8 is a requirement of the underlying [OpenTAD](https://github.com/sming256/OpenTAD) library itself.
`start.sh` installs Python dependencies, builds [decord](https://github.com/dmlc/decord) with CUDA, and compiles the custom CUDA sparse-conv extension (`opentad_sparse_ops`).

| | | | |
|---|---|---|---|
| `torch==2.0.1` | `torchvision==0.15.2` | `deepspeed` | `openmim` |
| `mmcv==2.0.1` | `mmaction2==1.1.0` | `flash-attn==2.5.4` | `images-framework` |
| `torchinfo` | `timm==1.0.24` | `mmengine` | `wandb` |
| `scipy` | `einops` | `pandas` | `tqdm` |
| `ninja` | `imgaug` | `pytorchvideo` | `numpy==1.23.5` |
| `gdown==5.1.0` | | | |

#### Usage
```
usage: eccv26_tad_test.py [-h] [--input-data INPUT_DATA] [--thresh THRESH] [--topk TOPK] [--data-root DATA_ROOT] [--save-video]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --thresh option to only show predictions with score above this threshold.

* Use the --topk option to show at most this many predictions (sorted by score).

* Use the --data-root option to override the raw video / feature data root path.

* Use the --save-video option to save the processed video.
```
usage: Recognition --database DATABASE
```

* Use the --database option to select the database model.
```
usage: ECCV26TAD [--gpu GPU] [--config FILE] [--ckpt CKPT]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).

* Use the --config option to set the path to config file.

* Use the --ckpt option to set the checkpoint path.
```
> torchrun --nnodes=1 --nproc_per_node=1 test/eccv26_tad_test.py --input-data test/example.mp4 --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --ckpt data/thumos_vitb.pth --topk 10 --database thumos --gpu 0 --data-root test/ --save-video
```

#### SparseConv2D CUDA Library
The custom CUDA kernels in `opentad/models/bricks/` implement the forward and backward passes of our native sparse 2D convolution. They are compiled as part of `pip install -e .` and importable as:

```python
import opentad_sparse_ops
```

The kernels are also usable standalone. The neighbor index table construction and sparse forward/backward passes are exposed directly, making the primitive reusable in any ViT adapter or FPN that follows token selection.

**Crossover point:** SparseConv2D matches dense convolution speed at ~55% keep rate and achieves up to 8× speedup at 10% keep rate (see paper Fig. 3). The crossover is on the adapter bottleneck channels, after a 4× channel reduction.

#### Architecture
SV-TAD is built on [OpenTAD](https://github.com/sming256/OpenTAD). The novel components are:

- **`opentad/models/backbones/vit_sparse_adapter_poguise.py`** — `VisionTransformerSparseAdapterPOGUISE`, the main SV-TAD backbone with sparse conv adapters and attention-based token selection.
- **`opentad/models/bricks/sparse_conv_layer.py`** — Python wrapper around the CUDA kernels; constructs the neighbor index table and dispatches sparse forward/backward.
- **`opentad/models/bricks/2d_sparse_v3.cu`, `implicit_gemm.cu`, `implicit_gemm_v2.cu`** — CUDA kernel implementations (tiled gather-GEMM for small channels; cuBLAS implicit GEMM for large channels).

#### Acknowledgements
SV-TAD is built on top of [OpenTAD](https://github.com/sming256/OpenTAD). We thank the OpenTAD authors for their open-source framework and pre-extracted features.