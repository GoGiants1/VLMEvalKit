uv venv --seed --python 3.10
source .venv/bin/activate

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
uv pip install -e .
uv pip install -e ".[train]"

cd ..
uv pip install -e .
MAX_JOBS=4 uv pip install "flash-attn<=2.7.4.post1" --no-build-isolation
