# Capstone Stable Audio Distillation

This project demonstrates a minimal training script that performs knowledge distillation on audio models using the [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) library.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The script expects `torch`, `torchaudio`, and `stable-audio-tools` to be available.

## Usage

Run the training script by specifying a dataset directory and teacher checkpoint:

```bash
python train.py \
    --dataset /path/to/librispeech \
    --teacher /path/to/teacher.ckpt \
    --epochs 5 \
    --batch-size 4 \
    --output-dir checkpoints
```

The distilled student checkpoints will be saved in the `--output-dir` directory.

## Testing

Basic unit tests are provided and can be run with:

```bash
pytest -q
```
