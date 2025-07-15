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
@click.option("--model", default="Llama-3.1-8B-Instruct",  help="Which model config to load")
@click.option("--experiment", default="unlearn/tofu/default", help="Experiment to use")
@click.option("--forget-split", default=None, help="Forget split")
@click.option("--retain-split", default=None, help="Retain split")
@click.option("--trainer", default="NPO", help="Unlearning Method")
@click.option("--paths-output-dir",required = True,  help="Override for paths.output_dir")
@click.option("--task-name", required=True, help="Name to tag this unlearning run")

def unlearn_tofu(config_name,model,experiment, forget_split, retain_split, trainer, paths_output_dir, task_name):
    """Run TOFU unlearning"""
    click.echo(f" Working directory: {TOFU_DIR}")
    command = [

        "python", "src/train.py",
        f"--config-name={config_name}",
        f"model={model}",
        f"experiment={experiment}",
        f"forget_split={forget_split}",
        f"retain_split={retain_split}",
        f"trainer={trainer}",
        f"paths.output_dir= /shared/share_mala/jt3585/newModels/{paths_output_dir}",
        f"task_name={task_name}",

       

    ]
    subprocess.run(command, cwd=TOFU_DIR)

@cli.command()
@click.option("--config-name", default="unlearn.yaml", help="Config to use")
@click.option("--model", default="Llama-3.1-8B-Instruct",  help="Which model config to load")
@click.option("--experiment", default="unlearn/tofu/default", help="Experiment to use")
@click.option("--trainer", default="NPO", help="Unlearning Method")
@click.option("--paths-output-dir",required = True,  help="Override for paths.output_dir")
@click.option("--task-name", required=True, help="Name to tag this unlearning run")

def train_tofu(config_name,model,experiment, trainer, paths_output_dir, task_name):
    """Run TOFU unlearning"""
    click.echo(f" Working directory: {TOFU_DIR}")
    command = [

        "python", "src/train.py",
        f"--config-name={config_name}",
        f"model={model}",
        f"experiment={experiment}",
        f"trainer={trainer}",
        f"paths.output_dir= /shared/share_mala/jt3585/newModels/{paths_output_dir}",
        f"task_name={task_name}",

       

    ]
    subprocess.run(command, cwd=TOFU_DIR)

@cli.command()
@click.option("--config-name", required=True, help="Config to use")
@click.option("--model", default="Llama-3.1-8B-Instruct",  help="Which model config to load")
@click.option("--experiment", default="eval/tofu/llama2", help="Experiment to use")
@click.option("--pretrained-model", help="Pretrained model")
@click.option("--paths-output-dir",required = True,  help="Override for paths.output_dir")
def evaluate_tofu(config_name, model, experiment, pretrained_model, paths_output_dir):
    """Evaluate model on TOFU benchmark"""
    click.echo("Evaluating model (from YAML config)")
    command = [
        "python", "src/eval.py",
        f"--config-name={config_name}",
        f"model={model}",
        f"experiment={experiment}",
        f"model.model_args.pretrained_model_name_or_path={pretrained_model}",
        f"paths.output_dir=/user/jt3585/unlearn/blackBox/answerGen/modelLogs/{paths_output_dir}",
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
