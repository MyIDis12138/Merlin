

### pre-commit configurations

```
pre-commit install
pre-commit run --all-files
```


### dependencies installation

```
pip install -e . --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
```

uv install [recommended]
```
cd ECE6780_project_group5
uv venv --python 3.10
source .venv/bin/activate
# source .venv/bin/activate.fish
uv pip install -e .[dev]
```
