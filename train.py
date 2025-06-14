import argparse
import os

try:
    import torch
    import torchaudio
except Exception:  # pragma: no cover - dependencies might be missing during tests
    torch = None
    torchaudio = None

try:
    import stable_audio_tools as sat
except ImportError:  # pragma: no cover - stable-audio-tools may not be installed
    sat = None


base_class = torch.nn.Module if torch else object


class StudentModel(base_class):
    """Simple student model using stable-audio-tools components if available."""

    def __init__(self):
        if torch is None:
            raise ImportError("PyTorch is required to use StudentModel")

        super().__init__()
        if sat and hasattr(sat, "Model"):
            self.model = sat.Model()
        else:  # fallback simple network
            self.model = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(16, 1, kernel_size=3, padding=1),
            )

    def forward(self, x):
        return self.model(x)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Stable Audio Knowledge Distillation")
    parser.add_argument("--dataset", required=True, help="Path to training dataset")
    parser.add_argument("--teacher", required=True, help="Path to teacher model checkpoint")
    parser.add_argument("--student", default=None, help="Optional path to student checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory to save checkpoints")
    return parser.parse_args(argv)


def load_teacher(path: str):
    if sat and hasattr(sat, "load_model"):
        return sat.load_model(path)
    # Fallback dummy teacher
    model = StudentModel()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model


def main(argv=None):
    if torch is None or torchaudio is None:
        raise ImportError("PyTorch and torchaudio are required for training")
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    teacher = load_teacher(args.teacher)
    student = StudentModel()
    if args.student and os.path.exists(args.student):
        student.load_state_dict(torch.load(args.student))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters())

    dataset = torchaudio.datasets.LIBRISPEECH(
        args.dataset, url="train-clean-100", download=False
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    teacher.eval()
    for epoch in range(args.epochs):
        student.train()
        for waveforms, _, _, _, _ in loader:
            with torch.no_grad():
                teacher_out = teacher(waveforms)
            student_out = student(waveforms)
            loss = criterion(student_out, teacher_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ckpt = os.path.join(args.output_dir, f"student_epoch_{epoch+1}.pt")
        torch.save(student.state_dict(), ckpt)

    return student


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
