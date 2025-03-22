## Project configuration
```
git clone git@github.com:MyIDis12138/ECE6780_project_group5.git
cd ECE6780_project_group5
ln -s /storage/ice1/shared/bmed6780/mip_group_5 data
```

### dependencies installation

```
pip install -e . --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
```

uv install [recommended]
```
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ECE6780_project_group5
uv venv --python 3.10
source .venv/bin/activate
# source .venv/bin/activate.fish
uv pip install -e .[dev]
```


### Experiment script

```
source .venv/bin/activate
python src/run.py configs/config.yaml
```
