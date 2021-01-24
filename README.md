## How to begin:

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

## How to train models:

1. Download FMA small dataset (8k tracks, 30s each) from https://os.unil.cloud.switch.ch/fma/fma_small.zip and unzip in
   ./data
2. `python build_dataset.py` (this will take a lot of time, but can be stopped and resumed at any moment)
3. `python train.py` (see `python train.py --help` for options, can be stopped and resumed)
4. (optional -
   evaluation) `python test.py --small_encoder <path to small encoder state> --large_encoder <path to large encoder state>`

### ...or download pre-trained models from [here](https://drive.google.com/drive/folders/1svcrICFPx_Awd6jS60uXFjZ0Ds9j1IQ1?usp=sharing).

## How to index your music database:

`python indexer.py --small_encoder <path to small encoder state> --large_encoder <path to large encoder state> --database <where to save track database> --index <where to save lookup index> --data <directory containing your mp3 files>` (
this will take a lot of time)

## How to recognize audio files:

Use recognition.py (see `python recogonition.py --help` for options).

This tool works either in an interactive mode (audio sample paths, offsets and lengths provided via stdin) or can be
pointed to a directory containing mp3 files (random samples will be taken). Output the name of the closest matching
track in the database (and similarity score - the lower the better).

## TODO:
* Recognize audio samples directly from pulseaudio.
* Split recognition.py into a client and a server (data loads slowly).
