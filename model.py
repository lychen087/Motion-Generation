import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class ConvEmbedding(nn.Module):
    """
    Define data embedding layer
    """
    def __init__(self, in_channels, out_channels, maxlen=640, kernel_size=11, stride=1):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size // 2
        )
        self.pos_embedding = nn.Embedding(maxlen, out_channels)
        self.maxlen = maxlen

    def forward(self, x):
        # x: [batch, time, feature]
        batch_size, time_steps, feature_dim = x.size()
        x = x.permute(0, 2, 1)  # [batch, feature, time] for Conv1d
        x = self.conv(x)  # [batch, feature, time]
        x = x.permute(0, 2, 1)  # [batch, time, feature] for transformer compatibility
        
        # Positional embedding
        position_ids = torch.arange(time_steps, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_embedding = self.pos_embedding(position_ids)  # [batch, time, feature]
        # print("position_embedding shape:", position_embedding.shape)
        return x + position_embedding

class MotionGenerationTransformer(nn.Module):
    def __init__(
            self, 
            audio_feature_dim, 
            label_feature_dim, 
            time_frame_size, 
            audio_embed_dim, 
            label_embed_dim, 
            hidden_dim,   
            ff_dim, 
            num_heads,
            num_layers_main, 
            num_layers_branch, 
            output_dims,
            dropout_rate=0.1
        ):

        super(MotionGenerationTransformer, self).__init__()
        
        # Input embedding layers
        self.audio_embedding = ConvEmbedding(audio_feature_dim, audio_embed_dim, time_frame_size)
        self.feature_embedding = ConvEmbedding(label_feature_dim, label_embed_dim, time_frame_size)

        # Positional embedding
        # self.position_embedding = nn.Embedding(time_frame_size, hidden_dim)
        
        # Main encoder
        self.main_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim + label_embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu"
            ) for _ in range(num_layers_main)
        ])
        
        # Branch encoders
        self.rhand_branch = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim + label_embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu"
            ) for _ in range(num_layers_branch)
        ])
        self.lhand_branch = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim + label_embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu"
            ) for _ in range(num_layers_branch)
        ])
        self.head_branch = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim + label_embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu"
            ) for _ in range(num_layers_branch)
        ])
        self.torso_branch = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim + label_embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu"
            ) for _ in range(num_layers_branch)
        ])
        self.root_branch = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim + label_embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu"
            ) for _ in range(num_layers_branch)
        ])
        self.bridge_branch = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim + label_embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu"
            ) for _ in range(num_layers_branch)
        ])
        self.volute_branch = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim + label_embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu"
            ) for _ in range(num_layers_branch)
        ])
        
        # Output layers
        self.rhand_output = nn.Linear(audio_embed_dim + label_embed_dim, output_dims["rhand"])
        self.lhand_output = nn.Linear(audio_embed_dim + label_embed_dim, output_dims["lhand"])
        self.head_output = nn.Linear(audio_embed_dim + label_embed_dim, output_dims["head"])
        self.torso_output = nn.Linear(audio_embed_dim + label_embed_dim, output_dims["torso"])
        self.root_output = nn.Linear(audio_embed_dim + label_embed_dim, output_dims["root"])
        self.bridge_output = nn.Linear(audio_embed_dim + label_embed_dim, output_dims["bridge"])
        self.volute_output = nn.Linear(audio_embed_dim + label_embed_dim, output_dims["volute"])


    def forward(self, audio_input, feature_input):
        # Embedding
        audio_embedding = self.audio_embedding(audio_input)  # [batch, time, embed_dim]
        feature_embedding = self.feature_embedding(feature_input)  # [batch, time, embed_dim]
        total_embedding = torch.cat((audio_embedding, feature_embedding), dim=-1)

        print("audio_embed shape:", audio_embedding.shape)  # 应为 [batch_size, time_frame_size, audio_embed_dim]
        print("feature_embed shape:", feature_embedding.shape)  # 应为 [batch_size, time_frame_size, label_embed_dim]
        # print("position_embed shape:", position_embedding.shape)  # 应为 [1, time_frame_size, hidden_dim]
        print("Total embedding shape:", total_embedding.shape)  # 应为 [batch_size, time_frame_size, audio_embed_dim + label_embed_dim]

        # Main encoder
        main_branch = total_embedding
        for layer in self.main_encoder:
            main_branch = layer(main_branch)
        
        # Branch encoders
        rhand = main_branch
        for layer in self.rhand_branch:
            rhand = layer(rhand)
        rhand_output = self.rhand_output(rhand)
        
        lhand = main_branch
        for layer in self.lhand_branch:
            lhand = layer(lhand)
        lhand_output = self.lhand_output(lhand)
        
        head = main_branch
        for layer in self.head_branch:
            head = layer(head)
        head_output = self.head_output(head)
        
        torso = main_branch
        for layer in self.torso_branch:
            torso = layer(torso)
        torso_output = self.torso_output(torso)
        
        root = main_branch
        for layer in self.root_branch:
            root = layer(root)
        root_output = self.root_output(root)

        bridge = main_branch
        for layer in self.bridge_branch:
            bridge = layer(bridge)
        bridge_output = self.bridge_output(bridge)

        volute = main_branch
        for layer in self.volute_branch:
            volute = layer(volute)
        volute_output = self.volute_output(volute)
        
        return rhand_output, lhand_output, head_output, torso_output, root_output, bridge_output, volute_output