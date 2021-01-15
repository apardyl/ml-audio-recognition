# How to run:
1. `python -m venv venv` 
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`   
4. Download FMA small dataset (8k tracks, 30s each) from https://os.unil.cloud.switch.ch/fma/fma_small.zip and unzip in ./data
5. `python build_dataset.py` (this will take a lot of time, but can be stopped and resumed at any moment)

