from pathlib import Path
import typer
import wandb

from commonlit_summaries.data import PromptType, PredictionType, SummaryDataset, load_data
from commonlit_summaries.experiment import Experiment, get_lr_scheduler, get_optimizer
from commonlit_summaries.losses import get_loss_fn
from commonlit_summaries.models import get_model
from commonlit_summaries.utils import set_seed
from commonlit_summaries.tokenizer import setup_tokenizer


app = typer.Typer(add_completion=False)


@app.command()
def main(
    group_id: int = typer.Option(..., "--group-id"),
    prompt_types: list[PromptType] = typer.Option(..., "--prompt-type"),
    prediction_type: PredictionType = typer.Option(..., "--prediction-type"),
    fold: str = typer.Option(..., "--fold"),
    model_name: str = typer.Option(..., "--name"),
    model_checkpoint: str = typer.Option(..., "--checkpoint"),
    data_dir: Path = typer.Option("./", "--data-dir"),
    max_length: int = typer.Option(512, "--max-length"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    train_batch_size: int = typer.Option(32, "--train-batch-size"),
    valid_batch_size: int = typer.Option(128, "--valid-batch-size"),
    epochs: int = typer.Option(1, "--epochs"),
    seed: int = typer.Option(666, "--seed"),
    scheduler_name: str = typer.Option("constant", "--scheduler"),
    warmup: float = typer.Option(0.0, "--warmup"),
    save_dir: Path = typer.Option("./", "--save-dir"),
    save_strategy: str = typer.Option("last", "--save-strategy"),
    accumulation_steps: int = typer.Option(1, "--accumulation-steps"),
    loss: str = typer.Option("mse", "--loss"),
    log_interval: int = typer.Option(100, "--log-interval"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    summariser_checkpoint: str = typer.Option("facebook/bart-large-cnn", "--summariser-checkpoint"),
    summariser_max_length: int = typer.Option(1024, "--summariser-max-length"),
    summariser_min_length: int = typer.Option(1024, "--summariser-min-length"),
    pooler: str = typer.Option("mean", "--pooler"),
    use_attention_head: bool = typer.Option(False, "--use-attention-head"),
    hidden_dropout_prob: float = typer.Option(0.1, "--hidden-dropout"),
    attention_probs_dropout_prob: float = typer.Option(0.1, "--attention-dropout"),
):
    wandb.login()
    wandb.init(
        project="commonlit-summaries",
        group=str(group_id),
        name=f"{group_id}-{model_name}-{fold}",
        config=locals(),
    )
    set_seed(seed)

    data = load_data(
        data_dir,
        train=True,
        summarise=PromptType.reference_summary in prompt_types,
        checkpoint=summariser_checkpoint,
        max_length=summariser_max_length,
        min_length=summariser_min_length,
    )

    print(f"Configuring inputs as: {[p.value for p in prompt_types]}")
    tokenizer = setup_tokenizer(model_checkpoint)

    if fold == "all":
        train_dataset = SummaryDataset(
            tokenizer, data, prompt_types, prediction_type, fix_length=max_length
        )
        valid_dataset = None
    else:
        train_data = data.loc[data.prompt_id != fold].reset_index(drop=True)
        valid_data = data.loc[data.prompt_id == fold].reset_index(drop=True)
        train_dataset = SummaryDataset(
            tokenizer, train_data, prompt_types, prediction_type, fix_length=max_length
        )
        valid_dataset = SummaryDataset(
            tokenizer, valid_data, prompt_types, prediction_type, fix_length=max_length
        )

    num_labels = 2 if prediction_type == PredictionType.both else 1
    loss_fn, metrics = get_loss_fn(loss, num_labels)
    model = get_model(
        model_checkpoint,
        num_labels,
        tokenizer_embedding_size=len(tokenizer),
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        pooler=pooler,
        use_attention_head=use_attention_head,
        device="cuda",
    )
    optimizer = get_optimizer(model, learning_rate, weight_decay)
    epoch_steps = (len(train_dataset) // train_batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(scheduler_name, optimizer, warmup, epochs, epoch_steps)
    experiment = Experiment(
        run_id=group_id,
        fold=fold,
        loss_fn=loss_fn,
        model_name=model_name,
        model=model,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        train_batch_size=train_batch_size,
        eval_batch_size=valid_batch_size,
        epochs=epochs,
        save_dir=save_dir,
        accumulation_steps=accumulation_steps,
        save_strategy=save_strategy,
        log_interval=log_interval,
        use_wandb=True,
    )
    experiment.run()


if __name__ == "__main__":
    app()
