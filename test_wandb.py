import wandb

wandb.init(project="test-project")
wandb.log({"test_metric": 123})
wandb.finish()
