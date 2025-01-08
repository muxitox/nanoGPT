salloc -n 1 -p gpu --gres=gpu -t 02:00:00 

module load Python/3.9.5-GCCcore-10.3.0 CUDA/11.3.1 GCC/10.3.0

source venv/bin/activate

# CUDA 11.0 https://pytorch.org/get-started/previous-versions/
python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/cu110/torch_stable.html

