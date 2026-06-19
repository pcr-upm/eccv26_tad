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
usage: ECCV26TAD [--gpu GPU] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--patience PATIENCE]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).
```
> python test/eccv26_tad_test.py --input-data test/example.jpg --database affectnet --gpu 0 --save-image
```
