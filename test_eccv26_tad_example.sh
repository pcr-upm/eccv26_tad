#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t eccv26_tad_image .
sudo docker run --name eccv26_tad_container --ipc=host --rm --gpus all -it -d eccv26_tad_image bash
#sudo docker run --name eccv26_tad_container -v /media/bobetocalo/database/classification/humans:/datasets --ipc host --rm --gpus all -it eccv26_tad_image bash
sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container bash start.sh
sudo docker exec -e WANDB_MODE=disabled -w /home/username/eccv26_tad eccv26_tad_container torchrun --nnodes=1 --nproc_per_node=1 test/eccv26_tad_test.py --input-data test/example.mp4 --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --ckpt data/thumos_vitb.pth --topk 10 --database thumos --gpu 0 --data-root test/ --save-video
#sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 test/eccv26_tad_train.py --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --wandb --database thumos
#sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 test/eccv26_tad_database.py --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --ckpt data/thumos_vitb.pth --class-map /datasets/THUMOS14/annotations/category_idx.txt --ann-file /datasets/THUMOS14/annotations/thumos_14_anno.json --data-root /datasets/THUMOS14/raw_data/video/ --database thumos
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp eccv26_tad_container:/home/username/conda/envs/eccv26/lib/python3.10/site-packages/images_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f eccv26_tad_container
sudo docker image rm eccv26_tad_image
sudo docker builder prune -a -f