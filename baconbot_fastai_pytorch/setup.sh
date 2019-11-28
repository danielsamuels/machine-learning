if [[ ! -d .venv ]]; then
  /c/Program\ Files/Python37/python.exe -m venv .venv
fi

. .venv/Scripts/activate
python -m pip install --upgrade pip setuptools
python -m pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install https://download.lfd.uci.edu/pythonlibs/t7epjj8p/Bottleneck-1.3.1-cp37-cp37m-win_amd64.whl
python -m pip install -r requirements.txt

python -c 'import torch; assert torch.cuda.is_available()'
python -m fastai.utils.show_install
