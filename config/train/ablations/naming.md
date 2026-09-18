# Ablation names

| name | what it means |
| --- | --- |
| `full` | every rendering effect on, in-air data (FlyingThings) included |
| `no_inair` | same, but simulated underwater data only |
| `haze_only` | water column only: no caustics, directional light, particles or halo |
| `*_pretrain` | trained without the warping loss, so no real-world data |
| `warp_finetune_*` | the matching `*_pretrain` run, fine-tuned with the warping loss on real data |

Every warping run is a fine-tune, so `warp_finetune_full` resumes `full_pretrain`.
