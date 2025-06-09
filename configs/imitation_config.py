import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Dataset configuration
    config.dataset = ml_collections.ConfigDict()
    config.dataset.data_dir = "/dataset"  # This matches your Modal volume mount
    config.dataset.batch_size = 64
    config.dataset.shuffle_buffer_size = 10000
    
    # Model configuration
    config.model = ml_collections.ConfigDict()
    config.model.hidden_size = 512
    config.model.num_layers = 3
    config.model.dropout_rate = 0.1
    
    # Training configuration
    config.training = ml_collections.ConfigDict()
    config.training.learning_rate = 3e-4
    config.training.num_epochs = 100
    config.training.warmup_steps = 1000
    config.training.eval_every = 1000
    config.training.save_every = 5000
    
    # Logging configuration
    config.logging = ml_collections.ConfigDict()
    config.logging.log_every = 100
    config.logging.wandb_project = "slippi-ai-training"
    
    # Paths
    config.paths = ml_collections.ConfigDict()
    config.paths.checkpoint_dir = "/models"  # This matches your Modal volume mount
    config.paths.log_dir = "/models/logs"
    
    return config