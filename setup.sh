. .venv/Scripts/activate
python -m pip install --upgrade pip
pip install tensorflow-gpu pyyaml h5py numpy Pillow scipy --ignore-installed
pip install jupyterlab jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
