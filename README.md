# g09-p2
The src dir has the "main" model which is a CVAE model. This is the one that is used for inference in the Docker container. Outside the src dir we have two more models that we also experimented with. 

# Setup env
Make sure to use Python 3.10 to match the pytorch image used in the docker image. 

### Setup env (using bash)

```sh
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### Setup env (using PowerShell)

```powershell
python3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements.txt
```
