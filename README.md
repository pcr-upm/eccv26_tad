# SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection

If you use this code for your own research, you must reference our conference paper:

```
SV-TAD: Native Sparse Convolutions for Efficient Temporal Action Detection 
Ricardo Pizarro, Roberto Valle, José M. Buenaposada, Luis M. Bergasa, Luis Baumela.
Proc. European Conference on Computer Vision, ECCV 2026.
```

#### Requisites
- images-framework

#### Usage
```
usage: eccv26_tad_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: ECCV26TAD [--gpu GPU] [--config FILE] [--ckpt CKPT] [--thresh THRESH] [--topk TOPK]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).

* Use the --config option to set the path to config file.

* Use the --ckpt option to set the checkpoint path.

* Use the --thresh option to only show predictions with score above this threshold.

* Use the --topk option to show at most this many predictions (sorted by score).
```
> python test/eccv26_tad_test.py --input-data test/example.mp4 --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --ckpt data/vitb_thumos_best.pth --topk 10 --database thumos --gpu 0
```
