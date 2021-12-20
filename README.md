# Neural Vocoder

HiFiGAN implementation. DLA CS HSE 2021 4th homework by Kostya Elenik.

## Instalation guide
```console
git clone https://github.com/kostyayatsok/NeuralVocoder.git
cd NeuralVocoder
pip install -qr requirements.txt
```
## Train
```console
python3 train.py -c configs/default_config.json
```

## Predict
```console
python3 test.py -o path/to/save/output -i path/with/original/wavs
```

## Wandb
- report: https://wandb.ai/kostyayatsok/hifi/reports/Homework-4-report--VmlldzoxMzU4MjI4
- all runs: https://wandb.ai/kostyayatsok/hifi?workspace=user-kostyayatsok
- best run: https://wandb.ai/kostyayatsok/hifi/runs/39lck09k
