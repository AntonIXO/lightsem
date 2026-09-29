cd /root/lightsem
VIRTUAL_ENV=/mnt/lightsem-data/venv-torch UV_CACHE_DIR=/mnt/lightsem-data/uv-cache uv pip install nnAudio transformers 2>&1 | tail -2
bash run_torch.sh -c "
from huggingface_hub import snapshot_download
p = snapshot_download('m-a-p/MERT-v1-330M', cache_dir='/mnt/lightsem-data/models/hf', allow_patterns=['*.json','*.py','*.bin','*.safetensors','*.txt'])
print('MERT at', p)
"
echo DL_MERT_DONE
