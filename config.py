config = {
    # "input_data": {
    #     "motion_normalize_range": 50,
    #     "audio_feature_size": 128,
    #     "label_feature_size": 10,
    #     "time_frame_size": 640
    # },
    "embedding": {
        "kernal_size": 11,
        "audio_embed_dim": 64,
        "label_embed_dim": 16,
        "num_heads": 4,
        "rhand_output_dim": 12,
        "lhand_output_dim": 12,
        "head_output_dim": 9,
        "embed_dim": 80
    },
    "model": {
        "dropout_rate": 0.1,
        "output_dim": 1,
        "num_layers_main_enc": 3,
        "num_layers_branch_enc": 3,
        "ff_dim": 320,
        "f_map": 16
    },
    "loss_weights": {
        "mse_loss_weight": 1,
        "mae_loss_weight": 0,
        "time_loss_weight": 2,
        "dim_loss_weight": 2,
        "rhand_loss_weight": 1,
        "lhand_loss_weight": 3,
        "head_loss_weight": 3
    },
    "training": {
        "batch_size": 16,
        "epoch_num": 200,
        "save_model_epoch": 10,
        "validation_split": 0.1,
        "initial_learning_rate": 2e-3
    }
}