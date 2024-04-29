step 1: clone Llava
step 2: git clone https://github.com/Dao-AILab/flash-attention.git
step 3: conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
step 4: pip install -e .
step 5: pip install -e ".[train]"
step 6: in flash attention folder, run: python setup.py install
