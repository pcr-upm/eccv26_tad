#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t eccv26_tad_image .
sudo docker volume create --name eccv26_tad_volume
sudo docker run --name eccv26_tad_container -v eccv26_tad_volume:/home/username --ipc=host --rm --gpus all -it -d eccv26_tad_image bash
#sudo docker run --name eccv26_tad_container -v /media/bobetocalo/database/classification/humans:/datasets --ipc host --rm --gpus all -it eccv26_tad_image bash
sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container bash start.sh
sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container torchrun --nnodes=1 --nproc_per_node=1 test/eccv26_tad_test.py --input-data test/example.mp4 --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --ckpt data/vitb_thumos_best.pth --topk 10 --database thumos --gpu 0 --save-video
#sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 test/eccv26_tad_database.py --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --ckpt data/vitb_thumos_best.pth --class-map /datasets/THUMOS14/annotations/category_idx.txt --ann-file /datasets/THUMOS14/annotations/thumos_14_anno.json --data-root /datasets/THUMOS14/raw_data/video --database thumos
sudo docker stop eccv26_tad_container
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
