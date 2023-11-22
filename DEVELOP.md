# Developers and maintainers readme

## Prerequesits

python3 pip and virtualenv
```bash
sudo apt install python3-pip
sudo apt install python3-virtualenv
```

## Installation

```bash
virtualenv -p /usr/bin/python3.10 .venv && source .venv/bin/activate
```

```bash
python -m pip install pip-tools
python -m pip install black
python -m pip install isort
```

## Development

### Update the requirements

```bash
python -m piptools compile
```

### Formatting

```bash
black --line-length 79 --target-version py310 --experimental-string-processing .
isort --virtual-env .venv --python-version 310 --profile black --gitignore .
```
