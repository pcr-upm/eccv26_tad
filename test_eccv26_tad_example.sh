#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t eccv26_tad_image .
sudo docker volume create --name eccv26_tad_volume
sudo docker run --name eccv26_tad_container -v eccv26_tad_volume:/home/username --rm --gpus all -it -d eccv26_tad_image bash
sudo docker exec -w /home/username/eccv26_tad eccv26_tad_container python test/eccv26_tad_test.py --input-data test/example.jpg --database affectnet --gpu 0 --save-image
sudo docker stop eccv26_tad_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/eccv26_tad_volume/_data/conda/envs/eccv26/lib/python3.12/site-packages/images_framework/output/images/ output
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
