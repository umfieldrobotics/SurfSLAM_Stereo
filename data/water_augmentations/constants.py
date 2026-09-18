import random
import torch

class WaterColumnParameters:
    def __init__(self, device=None, dtype=None):
        self.veiling_light_dict = {
            "deep_sea": torch.tensor([0.0, 0.0, 0.28]),
            # "shallow_water": torch.tensor([0.05, 0.11, 0.7]),
            "akdeniz": torch.tensor([0.14, 0.3, 0.5]),
            "river": torch.tensor([0.294, 0.4, 0.263]),
            "mud": torch.tensor([0.259, 0.259, 0.024]),
            "mhl": torch.tensor([0.0, 0.3021, 0.239]),
            "murky": torch.tensor([0.275, 0.212, 0.071]),
        }

        self.backscatters = {
            "Type I": torch.tensor([0.905, 0.961, 0.982]),
            "Type IA": torch.tensor([0.804, 0.954, 0.975]),
            "Type IB": torch.tensor([0.830, 0.940, 0.968]),
            "Type II": torch.tensor([0.800, 0.925, 0.940]),
            "Type III": torch.tensor([0.750, 0.885, 0.890]),
            "Type 1": torch.tensor([0.750, 0.885, 0.875]),
            "Type 3": torch.tensor([0.710, 0.820, 0.800]),
            "Type 5": torch.tensor([0.670, 0.730, 0.670]),
            "Type 7": torch.tensor([0.620, 0.610, 0.590]),
            "Type 9": torch.tensor([0.550, 0.460, 0.290]),
        }

        self.target_saturation_pixel_value = 0.01
        
        if device is not None:
            self.to(device=device)
        if dtype is not None:
            self.to(dtype=dtype)

    def to(self, *args, **kwargs):
        self.veiling_light_dict = {k: v.to(*args, **kwargs) for k, v in self.veiling_light_dict.items()}
        self.backscatters = {k: v.to(*args, **kwargs) for k, v in self.backscatters.items()}
    
    def sample_veiling_light(self):
        key = random.choice(list(self.veiling_light_dict.keys()))
        return self.veiling_light_dict[key]

    def sample_backscatter(self):
        key = random.choice(list(self.backscatters.keys()))
        bs_val = random.uniform(0, 1.0)
        return self.backscatters[key] * bs_val
