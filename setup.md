# Installation (L4 GPU on Lightning)
1. Clone the repo
```bash
git clone https://github.com/verl-project/verl.git
cd verl
```
2. To create the container:
```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl verlai/verl:vllm012.latest sleep infinity
docker start verl
docker exec -it verl bash
```
3. To install verl (run inside the container)
```bash
cd /workspace/verl
pip install -e .
```
4. To save (run outside the container):
```bash
docker save -o verl_docker_backup.tar verlai/verl:vllm012.latest
```
5. To load (for future sessions)
```bash
docker load -i ../verl_docker_backup.tar
```
# Running
1. To run verl, do the following commands from the verl repo path
```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
bash main.sh
```