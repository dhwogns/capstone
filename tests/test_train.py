import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("train", os.path.join(ROOT, "train.py"))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
parse_args = train.parse_args


def test_parse_args_defaults():
    args = parse_args(['--dataset', 'data', '--teacher', 'teacher.pt'])
    assert args.dataset == 'data'
    assert args.teacher == 'teacher.pt'
    assert args.epochs == 1
    assert args.batch_size == 1
    assert args.output_dir == 'checkpoints'
