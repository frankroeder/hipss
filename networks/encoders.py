from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):

    def __init__(self,
                 cfg,
                 env_params,
                 input_size=None,
                 feature_size: int = 512,
                 output_layer=False,
                 simple=True,
                 observation_input=True):
        super(FeatureExtractor, self).__init__()
        hidden_size = cfg.hidden_size
        self.img_input = env_params['image_observation'] and observation_input
        if self.img_input:
            self.extractor = VisionEncoder(cfg, env_params, feature_size)
        else:
            if simple:
                self.extractor = nn.Linear(input_size, feature_size)
            else:
                self.extractor = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, feature_size),
                )
        self.output_fn = nn.Identity() if output_layer else nn.ReLU()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.img_input:
            if len(observation.shape) == 5:
                # B, FrameStack, W, H, Channel [1, 3, 84, 84, 3]
                # permute: B, FrameStack, Channel, W, H [1, 3, 3, 84, 84]
                observation = observation.permute(0, 1, -1, 2, 3)
                # reshape: B, C, W, H [1, 9, 84, 84]
                reshaped = observation.reshape(-1, observation.shape[1] * observation.shape[2], observation.shape[3],
                                               observation.shape[4])
            else:
                # B, W, H, Channel [1, 84, 84, 3]
                # permute: B, Channel, W, H,  [1, 3, 84, 84]
                reshaped = observation.permute(0, -1, 1, 2)
            obs_embedding = self.extractor(reshaped)
        else:
            obs_embedding = self.extractor(observation)
        return self.output_fn(obs_embedding)


class LanguageEncoder(nn.Module):

    def __init__(self, cfg, env_params):
        super(LanguageEncoder, self).__init__()
        self.one_hot = cfg.one_hot
        self.bidirectional = cfg.bidirectional
        self.embedding_size = cfg.embedding_size
        self.with_gru = cfg.with_gru
        self.hidden_size = cfg.hidden_size
        self.vocab = env_params['vocab']
        max_instruction_len = env_params['instruction']
        self.max_action = env_params['action_max']

        if not self.one_hot:
            self.lang_embedding = nn.Embedding(len(self.vocab), self.embedding_size, padding_idx=0) # idx 0 = <pad>

        if self.with_gru:
            input_size = len(self.vocab) if self.one_hot else self.embedding_size
            self.gru = nn.GRU(input_size,
                              self.embedding_size,
                              bidirectional=self.bidirectional,
                              dropout=cfg.dropout,
                              batch_first=True)
            self._output_size = self.embedding_size * (2 if self.bidirectional else 1)
        else:
            flat_input_size = len(
                self.vocab) * max_instruction_len if self.one_hot else self.embedding_size * max_instruction_len
            self.flat = nn.Flatten()
            self.fc = nn.Linear(flat_input_size, self.embedding_size)
            self._output_size = self.embedding_size
        self.output_fn = nn.Identity() #nn.ReLU()

    def forward(self, instruction):
        if self.one_hot:
            lang_emb = F.one_hot(instruction, num_classes=len(self.vocab)).float()
        else:
            lang_emb = self.lang_embedding(instruction)

        if self.with_gru:
            outputs, final_state = self.gru(lang_emb)
            if self.bidirectional:
                lang_ctx = torch.cat((final_state[0:final_state.size(0):2], final_state[1:final_state.size(0):2]),
                                     dim=2).view(instruction.size(0), -1)
            else:
                lang_ctx = outputs[:, -1]
        else:
            lang_ctx = self.fc(self.flat(lang_emb))

        return self.output_fn(lang_ctx)

    @property
    def output_size(self):
        return self._output_size


class VisionEncoder(nn.Module):
    """
    :param observation_space:
    :param features_size: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, cfg, env_params: Dict, feature_size: int, architecture='nature'):
        super(VisionEncoder, self).__init__()
        assert feature_size > 0
        self.features_dim = feature_size
        # We assume CxHxW images (channels first)
        n_input_channels = env_params['channels'] * env_params['framestack']
        if architecture == 'nature':
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        elif architecture == 'drqv2':
            #  drqv2
            self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 32, 3, stride=2), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())
        else:
            raise NotImplemented("Unknown cnn architecture")

        if cfg.cnn_augmentation:
            # from drqv2
            self.aug = RandomShiftsAug(4)
        else:
            self.aug = nn.Identity()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.randn(1, n_input_channels, 84, 84).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.aug(observations)
        observations = observations.float() / 255.0 - 0.5
        return self.linear(self.cnn(observations))


class RandomShiftsAug(nn.Module):

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
