#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t eccv26_tad_image .
sudo docker volume create --name eccv26_tad_volume
sudo docker run --name eccv26_tad_container -v eccv26_tad_volume:/home/username --rm --gpus all -it -d eccv26_tad_image bash
sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container bash start.sh
sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container python test/eccv26_tad_test.py configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --input-data test/example.mp4 --checkpoint data/vitb_thumos_best.pth --topk 10
sudo docker stop eccv26_tad_container
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
