import click
import subprocess
from pathlib import Path

TOFU_DIR = (Path(__file__).parent.parent / "open-unlearning").resolve()

@click.group()
def cli():
    """TOFU Benchmark CLI"""
    pass

@cli.command()
@click.option("--config-name", default="unlearn.yaml", help="Config to use")
@click.option("--pretrained", required=True, help = "model path")
@click.option("--experiment", default="unlearn/tofu/default", help="Experiment to use")
@click.option("--forget-split", default="forget05", help="Forget split")
@click.option("--retain-split", default="retain95", help="Retain split")
@click.option("--trainer", default="GradAscent", help="Unlearning Method")
def unlearn_tofu(config_name,pretrained, experiment, forget_split, retain_split, trainer):
    """Run TOFU unlearning"""
    click.echo("Running unlearning with config-defined taskname and model")
    command = [
        "python", "src/train.py",
        f"--config-name={config_name}",
        f"model.model_args.pretrained_model_name_or_path={pretrained}",
        f"experiment={experiment}",
        f"forget_split={forget_split}",
        f"retain_split={retain_split}",
        f"trainer={trainer}",
        f"model.model_args.attn_implementation=sdpa"
    ]
    subprocess.run(command, cwd=TOFU_DIR)


@cli.command()
@click.option("--config-name", default="train.yaml", help="Config to use")
@click.option("--experiment", default="finetune/tofu/default", help="Experiment to use")
def finetune_tofu(config_name, experiment):
    """Run TOFU unlearning"""
    click.echo("Running unlearning with config-defined taskname and model")
    command = [
        "python", "src/train.py",
        f"--config-name={config_name}",
        f"experiment={experiment}"
    ]
    subprocess.run(command, cwd=TOFU_DIR)

@cli.command()
@click.option("--config-name", required=True, help="Config to use")
@click.option("--experiment", default="eval/tofu/llama2", help="Experiment to use")
@click.option("--pretrained-model", help="Pretrained model")
def evaluate_tofu(config_name, experiment, pretrained_model):
    """Evaluate model on TOFU benchmark"""
    click.echo("Evaluating model (from YAML config)")
    command = [
        "python", "src/eval.py",
        f"--config-name={config_name}",
        f"experiment={experiment}",
        f"model.model_args.pretrained_model_name_or_path={pretrained_model}",
        f"model.model_args.attn_implementation=sdpa"
    ]
    subprocess.run(command, cwd=TOFU_DIR)

@cli.command()
@click.option("--config-name", required=True, help="Config to use")
@click.option("--experiment", default="eval/muse/llama2", help="Experiment to use")
@click.option("--pretrained-model", help="Pretrained model")
def evaluate_muse(config_name, experiment, pretrained_model):
    """Evaluate model on MUSE benchmark"""
    click.echo("Evaluating model (from YAML config)")
    command = [
        "python", "src/eval.py",
        f"--config-name={config_name}",
        f"experiment={experiment}",
        f"model.model_args.pretrained_model_name_or_path={pretrained_model}",
        f"model.model_args.attn_implementation=sdpa"
    ]
    subprocess.run(command, cwd=TOFU_DIR)

if __name__ == "__main__":
    cli()
