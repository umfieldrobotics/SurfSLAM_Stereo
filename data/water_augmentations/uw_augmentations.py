import json
import math
import random
import numpy as np
import cv2
import warnings
import glob
import os
from data.water_augmentations.utils import *
from data.water_augmentations.constants import WaterColumnParameters
import tarfile
from pathlib import Path
from dataclasses import dataclass

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str) -> str:
    """Resolve `path` against the repo root unless it is already absolute."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else REPO_ROOT / p)


TEXTURE_SUFFIXES = (".png", ".jpg", ".jpeg")


def find_textures(texture_dir: str | Path) -> list[str]:
    """Texture images in `texture_dir`, sorted so the bank order is deterministic."""
    texture_dir = Path(texture_dir)
    if not texture_dir.is_dir():
        return []
    return sorted(
        str(p) for p in texture_dir.iterdir() if p.suffix.lower() in TEXTURE_SUFFIXES
    )


def ensure_texture_lib(texture_dir: str) -> str:
    """
    Texture libraries ship as .tar.xz archives next to the directory they unpack
    into (they are too large to keep unpacked in git). Extract on first use.
    """
    texture_dir = Path(resolve_repo_path(texture_dir))
    if find_textures(texture_dir):
        return str(texture_dir)

    archive = texture_dir.with_suffix(".tar.xz")
    if not archive.exists():
        raise RuntimeError(
            f"No textures in {texture_dir} and no archive at {archive} to extract."
        )

    print(f"Extracting {archive} -> {texture_dir.parent} (first use only)")
    with tarfile.open(archive, "r:xz") as tar:
        try:
            tar.extractall(texture_dir.parent, filter="data")
        except TypeError:  # python < 3.10.12 has no extraction filters
            tar.extractall(texture_dir.parent)

    if not find_textures(texture_dir):
        raise RuntimeError(
            f"{archive} did not contain a {texture_dir.name}/ directory of textures."
        )
    return str(texture_dir)


@dataclass
class WaterAugAblationSettings:
    enable_caustics: bool = True
    enable_water_column: bool = True
    enable_directional_light: bool = True
    enable_particles: bool = True
    enable_halo: bool = True


class WaterAugmentations:
    """
    The ```config``` is a dictionary with the following content: \n
        - caustic:
            - `caustic_texture_path`  : (optional)    Path to the caustic texture library, relative
                                                      to the repo root (extracted from its .tar.xz
                                                      on first use, see `ensure_texture_lib`).
            - `light_dir`             : (optional)    The direction that the caustics will be
                                                      amplified the most in, in the world frame.
            - `blend_weight`          : (optional)    Base blending weight (0 to 1.5) controlling
                                                      the maximum contribution of the caustics.
            - `normal_perturbation`   : (optional)    The maximum fraction to blend scene normals
                                                      toward the target direction.
            - `blend_spread`         : (optional)     The stddev of the Gaussian distribution
                                                      used to blend the caustics with the scene normals.
        - light:
            - `light_position_radius`       : (optional)      The radius around the left camera to sample the light source position.
            - `light_power_max`             : (optional)      The power of the light source.
            - `sun`:                        : (optional)      The color of the light source.
                - `enabled`                 : (optional)      Whether sun will be simulated.
                - `halo_probability`        : (optional)      Probability that the simulated sun will be a "halo" note: halo_probability + planar_prbability <= 0.
                - `planar_probability`      : (optional)      The probability of the light source being planar.
                - `transmission_end`        : (optional)      The end of the halo effect on the image.
        - water:
            - `attn_param`         : (optional)      Attenuation parameter for the light source.
            - `bs_param`           : (optional)      Background scattering parameter for the light source.
            - `veiling_light_param`: (optional)      Veiling light parameter for the light source.
            - `
        - augmentations:
            - `augmentations`       : (optional)      List of augmentations to apply to the RGB image.
        - max_depth: (optional)     : (optional) The maximum distance to consider when applying things

    The existance of any of these keys, even with a value of None, will apply the corresponding effect with randomized parameters.
    """

    def __init__(
        self, config_path: str, settings: WaterAugAblationSettings, device: str = "cpu"
    ):

        config_path = resolve_repo_path(config_path)
        with open(config_path, "r") as f:
            config = json.load(f)

        self.settings = settings
        self.aug_severity = 1.0

        self.cfg = config
        self.device = device

        self._current_backscatter_color = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device=device
        )

        self.water_column_params = WaterColumnParameters(device=device)

        self._printed_caustics = False
        self._printed_dir_light = False
        self._printed_water_col = False
        self._printed_particles = False

        caustics_dir = ensure_texture_lib(config["caustic"]["caustic_texture_path"])
        config["caustic"]["caustic_texture_path"] = caustics_dir
        files = find_textures(caustics_dir)
        self._caustic_cache = {}
        self._caustic_bank: list[torch.Tensor] = [
            cv2.cvtColor(cv2.imread(f, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            for f in files
        ]
        if not self._caustic_bank:
            raise RuntimeError(f"No caustic textures found in {caustics_dir}")

        particles_dir = ensure_texture_lib(config["particles"]["texture_path"])
        config["particles"]["texture_path"] = particles_dir
        files = find_textures(particles_dir)
        self._particles_cache = {}
        self._particles_bank: list[torch.Tensor] = [
            cv2.cvtColor(cv2.imread(f, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            for f in files
        ]
        if not self._particles_bank:
            raise RuntimeError(f"No particles textures found in {particles_dir}")

    def set_aug_severity(self, severity: float):
        self.aug_severity = max(0.0, min(1.0, severity))

    def __call__(
        self,
        stereo_info: dict,
    ):
        """
        Apply the augmentations to the RGB image and the effects to the depth image.
        Parameters:
            left_rgb             : HxWx3 RGB image left.
            right_rgb            : HxWx3 RGB image right.
            left_depth           : HxW depth image (per-pixel depth values).
            right_depth          : HxW depth image (per-pixel depth values).
            left_normals         : HxWx3 array of per-pixel normals (in the camera coordinate frame, assumed normalized).
            right_normals        : HxWx3 array of per-pixel normals (in the camera coordinate frame, assumed normalized).
            left_pose            : 4x4 camera pose matrix.
            cam_intrinsics_left  : 3x3 camera intrinsics matrix.
            cam_intrinsics_right : 3x3 camera intrinsics matrix.
            stereo_baseline      : baseline value (float)
            stereo_extrinsics    : 4x4 stereo extrinsics matrix. (left to right camera)
        """
        # left
        left_rgb = stereo_info.get("left_rgb", None)
        left_depth = stereo_info.get("left_depth", None)
        left_normals = stereo_info.get("left_normals", None)
        left_pts = stereo_info.get("left_pts", None)
        left_pose = stereo_info.get("left_pose", None)
        left_cam_intrinsics = stereo_info.get("cam_intrinsics_left", None)

        # right
        right_rgb = stereo_info.get("right_rgb", None)
        right_depth = stereo_info.get("right_depth", None)
        right_normals = stereo_info.get("right_normals", None)
        right_pts = stereo_info.get("right_pts", None)
        right_pose = stereo_info.get("right_pose", None)
        right_camera_intrinsics = stereo_info.get(
            "cam_intrinsics_right", left_cam_intrinsics
        )

        stereo_baseline = stereo_info.get("stereo_baseline", 0.25)

        augmented_image1 = left_rgb.permute(1, 2, 0).clone()
        augmented_image2 = right_rgb.permute(1, 2, 0).clone()

        attn_param, bs_param, veiling_light_param = None, None, None

        # power_lo, power_hi = sun_cfg.get("power_range", [10.0, 40.0])
        sun_cfg = self.cfg["light"]["sun"]

        base_color = torch.tensor([1.0, 0.94, 0.78], device=left_rgb.device)
        noise = torch.empty(3, device=left_rgb.device).normal_(mean=0.0, std=0.07)
        sun_color = torch.clamp(base_color + noise, min=0.7, max=1.0)

        sun_dir = torch.tensor(
            [np.random.rand() - 0.5, 1.0, np.random.rand() + 0.5],
            device=left_rgb.device,
        )
        sun_dir /= sun_dir.norm()

        self.sun_state = {
            "color": sun_color,
            "direction": sun_dir,
        }

        # The order matters -- each source of light should be attenuated as part of the image formation
        augmented_image1, augmented_image2 = self.apply_temporal_caustics(
            rgb_=augmented_image1,
            depth_=left_depth,
            normals_=left_normals,
            rgb2_=augmented_image2,
            depth2_=right_depth,
            normals2_=right_normals,
        )

        (
            augmented_image1,
            augmented_image2,
            attn_param,
            bs_param,
            veiling_light_param,
        ) = self.apply_light_source(
            rgb_=augmented_image1,
            rgb2_=augmented_image2,
            normals_=left_normals,
            normals2_=right_normals,
            points_=left_pts,
            points2_=right_pts,
            stereo_baseline_=stereo_baseline,
        )

        (augmented_image1, augmented_image2), (attn_map1, attn_map2) = (
            self.apply_water_column(
                rgb_=augmented_image1,
                depth_=left_depth,
                points_=left_pts,
                points2_=right_pts,
                rgb2_=augmented_image2,
                depth2_=right_depth,
                attn_param_=attn_param,
                bs_param_=bs_param,
                veiling_light_param_=veiling_light_param,
                return_attn_map=True,
            )
        )

        augmented_image1, augmented_image2 = self.apply_particles(
            rgb_=augmented_image1, rgb2_=augmented_image2
        )

        def f(x):
            return x.permute(2, 0, 1)

        return f(augmented_image1), f(augmented_image2), f(attn_map1), f(attn_map2)

    def apply_light_source(
        self,
        rgb_: torch.Tensor,
        rgb2_: torch.Tensor,
        normals_: torch.Tensor,
        normals2_: torch.Tensor,
        points_: torch.Tensor,
        points2_: torch.Tensor,
        stereo_baseline_: float = 0.25,
    ):

        # ---- point light -----------------------------------------------------------
        light_position_radius = self.cfg["light"].get("light_position_radius", 1.0)
        light_position = torch.randn(3, device=rgb_.device) * light_position_radius
        light_position2 = light_position.clone()
        light_position2[0] -= stereo_baseline_

        light_power_max = self.cfg["light"].get("light_power_max", 1.0)
        light_color = torch.tensor(
            self.cfg["light"].get("light_color", [1, 1, 1]), device=rgb_.device
        )
        ambient_light = self.cfg["light"].get("ambient_light", 0.0)

        attn_param = self.cfg["water"].get("attn_param", None)
        if attn_param is not None:
            attn_param = attn_param * self.aug_severity
        bs_param = self.cfg["water"].get("bs_param", None)
        if bs_param is not None:
            bs_param = bs_param * self.aug_severity
        veiling_light_param = self.cfg["water"].get("veiling_light_param", None)

        light_power = np.random.uniform(0, light_power_max)

        if attn_param is None:
            attn_param = self.water_column_params.sample_backscatter() * self.aug_severity
            bs_param = attn_param
        if veiling_light_param is None:
            veiling_light_param = self.water_column_params.sample_veiling_light()

        atten1 = attenuate_light_source(
            light_position,
            light_color,
            rgb_,
            points_,
            attn_param,
            bs_param,
            veiling_light_param,
        )
        atten2 = attenuate_light_source(
            light_position2,
            light_color,
            rgb2_,
            points2_,
            attn_param,
            bs_param,
            veiling_light_param,
        )

        rgb1 = simulate_light(
            light_position, 
            light_power, 
            atten1, rgb_, 
            normals_, 
            points_, 
            ambient_light,
            aug_severity=self.aug_severity,
        )
        rgb2 = simulate_light(
            light_position2,
            light_power,
            atten2,
            rgb2_,
            normals2_,
            points2_,
            ambient_light,
            aug_severity=self.aug_severity,
        )

        # ---- output gate -----------------------------------------------------------
        if self.settings.enable_directional_light:
            if not self._printed_dir_light:
                print("Applying directional light (point + depth-based sun)")
                self._printed_dir_light = True
            return rgb1, rgb2, attn_param, bs_param, veiling_light_param

        if not self._printed_dir_light:
            print("NOT applying directional light")
            self._printed_dir_light = True
        return rgb_, rgb2_, attn_param, bs_param, veiling_light_param

    def apply_temporal_caustics(
        self,
        rgb_: torch.Tensor,
        depth_: torch.Tensor,
        normals_: torch.Tensor = None,
        rgb2_: torch.Tensor = None,
        depth2_: torch.Tensor = None,
        normals2_: torch.Tensor = None,
    ):
        sun_dir = self.sun_state["direction"].to(rgb_.device)
        blend_weight = self.cfg["caustic"].get("blend_weight", 0.5)
        blend_spread = self.cfg["caustic"].get("blend_spread", 0.0)
        normal_perturbation = self.cfg["caustic"].get("normal_perturbation", None)

        caustic_texture_np = random.choice(self._caustic_bank)
        caustic_texture = torch.from_numpy(
            cv2.resize(
                caustic_texture_np,
                (rgb_.shape[1], rgb_.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        ).to(rgb_.device)

        rgb1, rgb2 = blend_temporal_caustics_with_geometry(
            rgb_=rgb_,
            depth_=depth_,
            normals_=normals_,
            rgb2_=rgb2_,
            depth2_=depth2_,
            normals2_=normals2_,
            caustic_texture_=caustic_texture,
            sun_dir_=sun_dir,
            blend_weight_=blend_weight,
            blend_spread_=blend_spread,
            normal_perturbation_=normal_perturbation,
            aug_severity=self.aug_severity,
        )

        if not self.settings.enable_caustics:
            if not self._printed_caustics:
                print("NOT Applying Caustics")
                self._printed_caustics = True
            return rgb_, rgb2_

        if not self._printed_caustics:
            print("Applying Caustics")
            self._printed_caustics = True

        return rgb1, rgb2

    def apply_caustics(
        self,
        rgb_: torch.Tensor,
        depth_: torch.Tensor,
        normals_: torch.Tensor = None,
    ):
        caustic_texture_path = self.cfg["caustic"].get("caustic_texture_path", None)
        sun_dir = self.cfg["caustic"].get("light_dir", [0.0, 1.0, 1.0])
        sun_dir = torch.tensor(sun_dir).to(rgb_.device)
        blend_weight = self.cfg["caustic"].get("blend_weight", 0.5)
        blend_spread = self.cfg["caustic"].get("blend_spread", 0.1)
        normal_perturbation = self.cfg["caustic"].get("normal_perturbation", None)

        caustic_textures_png = glob.glob(os.path.join(caustic_texture_path, "*.png"))
        caustic_textures_jpg = glob.glob(os.path.join(caustic_texture_path, "*.jpg"))
        caustic_textures_jpeg = glob.glob(os.path.join(caustic_texture_path, "*.jpeg"))

        caustic_textures = (
            caustic_textures_png + caustic_textures_jpg + caustic_textures_jpeg
        )
        # select a random caustic texture
        if len(caustic_textures) > 0:
            caustic_texture = cv2.imread(
                np.random.choice(caustic_textures), cv2.IMREAD_UNCHANGED
            )
            caustic_texture = cv2.cvtColor(caustic_texture, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(
                "No caustic texture found in the specified path. Please provide a valid path."
            )

        caustic_texture = torch.from_numpy(
            cv2.resize(
                caustic_texture,
                (rgb_.shape[1], rgb_.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        ).to(rgb_.device)

        rgb = blend_caustics_with_geometry(
            rgb_=rgb_,
            depth_=depth_,
            normals_=normals_,
            caustic_texture_=caustic_texture,
            sun_dir_=sun_dir,
            blend_weight_=blend_weight,
            blend_spread_=blend_spread,
            normal_perturbation_=normal_perturbation,
        )
        return rgb

    def apply_water_column(
        self,
        rgb_: torch.Tensor,
        points_: torch.Tensor,
        depth_: torch.Tensor,
        rgb2_: torch.Tensor = None,
        depth2_: torch.Tensor = None,
        points2_: torch.Tensor = None,
        attn_param_: torch.Tensor = None,
        bs_param_: torch.Tensor = None,
        veiling_light_param_: torch.Tensor = None,
        return_attn_map: bool = False,
        sun_color: torch.Tensor = torch.Tensor([1.0, 1.0, 1.0]),
    ):
        """
        Simulate the effect of a water column on the RGB image.

        Parameters:
            attn_params_            : Attenuation parameters for the light source. (scalar)
            bs_params_              : Background scattering parameters for the light source. (scalar)
            veiling_light_params_   : Veiling light parameters for the light source. (scalar)
        """
        # Prepare the inputs
        attn_param = (
            self.cfg["water"].get("attn_param", None)
            if attn_param_ is None
            else attn_param_
        )
        if attn_param is not None:
            attn_param = attn_param * self.aug_severity
        bs_param = (
            self.cfg["water"].get("bs_param", None) if bs_param_ is None else bs_param_
        )
        if bs_param is not None:
            bs_param = bs_param * self.aug_severity
        veiling_light_param = (
            self.cfg["water"].get("veiling_light_param", None)
            if veiling_light_param_ is None
            else veiling_light_param_
        )

        # If no user specifid parameters, sample from the default water column parameters
        if attn_param is None:
            attn_param = self.water_column_params.sample_backscatter() * self.aug_severity
            bs_param = attn_param
        if veiling_light_param is None:
            veiling_light_param = self.water_column_params.sample_veiling_light()
        if depth_.ndim == 2:
            depth_ = depth_.unsqueeze(-1)
        if depth2_ is not None and depth2_.ndim == 2:
            depth2_ = depth2_.unsqueeze(-1)

        if "max_depth" in self.cfg:
            depth_ = depth_.clamp(0, self.cfg["max_depth"])
            if depth2_ is not None:
                depth2_ = depth2_.clamp(0, self.cfg["max_depth"])

        lo = self.cfg["water"]["min_distance_pct"]
        hi = self.cfg["water"]["max_distance_pct"]
        percentage_sampler = np.random.rand() * (hi - lo) + lo

        d2 = depth2_.max() if depth2_ is not None else depth_.new_tensor(0.0)
        target_saturation_depth = max(depth_.max(), d2) * percentage_sampler

        # find the linear multiplier based on the target saturation depth and pixel to saturate to
        if self.aug_severity > 0:
            linear_multiplier = (
                -np.log(self.water_column_params.target_saturation_pixel_value)
                / (target_saturation_depth * attn_param.max())
                if target_saturation_depth > 3
                else 1.0
            )
        else:
            linear_multiplier = 1.0

        if not self.settings.enable_water_column:
            if not self._printed_water_col:
                print("NOT applying water column")
                self._printed_water_col = True
            if return_attn_map:
                attn_map = torch.ones_like(rgb_)
                if rgb2_ is not None and depth2_ is not None:
                    return (rgb_, rgb2_), (attn_map, attn_map)
                return rgb_, attn_map
            if rgb2_ is not None and depth2_ is not None:
                return rgb_, rgb2_

        if not self._printed_water_col:
            print("Applying water column")
            self._printed_water_col = True

        # Akkaynak et al. 2019
        attenuation_map = torch.exp(-depth_ * linear_multiplier * attn_param)
        backscatter_map = 1 - torch.exp(-depth_ * linear_multiplier * bs_param)
        final_image = (
            (rgb_.float() / 255.0) * attenuation_map
            + veiling_light_param * backscatter_map
        ) * 255.0
        final_image = final_image.clamp(0, 255).to(dtype=torch.uint8)

        self._current_backscatter_color = torch.as_tensor(
            (
                [veiling_light_param] * 3
                if not torch.is_tensor(veiling_light_param)
                else veiling_light_param
            ),
            device=rgb_.device,
            dtype=torch.float32,
        )

        if rgb2_ is not None and depth2_ is not None:
            attenuation_map2 = torch.exp(-depth2_ * linear_multiplier * attn_param)
            backscatter_map2 = 1 - torch.exp(-depth2_ * linear_multiplier * bs_param)
            final_image2 = (
                (rgb2_.float() / 255.0) * attenuation_map2
                + veiling_light_param * backscatter_map2
            ) * 255.0
            final_image2 = final_image2.clamp(0, 255).to(dtype=torch.uint8)

        sun_cfg: dict = self.cfg["light"].get("sun", None)

        if sun_cfg is not None and sun_cfg["enabled"]:
            halo_probability = sun_cfg.get("halo_probability", 0.0)
            planar_probability = sun_cfg.get("planar_probability", 0.0)
            sun_probability = sun_cfg.get("sun_probability", 0.0)
            assert (
                halo_probability + planar_probability + sun_probability <= 1.0
            ), "Halo and planar probabilities must sum to 1 or less."
            rand_num = random.random()
            H, W = rgb_.shape[:2]
            if rand_num < planar_probability:
                planar_cfg: dict = sun_cfg.get("planar", {})
                transmission_end_list = planar_cfg.get("transmission_end", [0.0, 1.0])
                max_influence_range = planar_cfg.get(
                    "max_light_influence_range", [0.5, 0.7]
                )
                max_influence = random.uniform(
                    max_influence_range[0], max_influence_range[1]
                )
                min_influence = planar_cfg.get("min_light_influence", 0.05)
                transmission_end = random.uniform(
                    transmission_end_list[0], transmission_end_list[1]
                )
                rows = torch.arange(H, device=final_image.device).view(-1, 1)
                gamma_ref = torch.log(
                    torch.tensor(max_influence, device=final_image.device)
                    / torch.tensor(min_influence, device=final_image.device)
                ) / (transmission_end * H)
                if attn_param.sum() == 0 and self.aug_severity == 0:
                    gamma_vec = torch.zeros_like(attn_param)
                else:
                    gamma_vec = gamma_ref * (attn_param / attn_param.max())
                attenuated_sun_profile = max_influence * torch.exp(-rows * gamma_vec)
                sun_img = torch.repeat_interleave(
                    attenuated_sun_profile[:, None, :], W, dim=1
                ) * self.sun_state["color"].to(final_image.device)

                if self.settings.enable_halo:
                    sunny_img1 = ((final_image / 255) + sun_img * self.aug_severity).clamp(0, 1) * 255
                    sunny_img2 = ((final_image2 / 255) + sun_img * self.aug_severity).clamp(0, 1) * 255
                    final_image = sunny_img1.to(final_image.device).type(torch.uint8)
                    final_image2 = sunny_img2.to(final_image2.device).type(torch.uint8)

            elif rand_num < halo_probability + planar_probability:
                halo_cfg = sun_cfg.get("halo", {})
                top_y = halo_cfg.get("light_source_top", -1.0)
                bottom_y = halo_cfg.get("light_source_bottom", -0.25)
                max_influence_range = halo_cfg.get(
                    "max_light_influence_range", [0.3, 0.7]
                )
                max_influence = random.uniform(
                    max_influence_range[0], max_influence_range[1]
                )
                device = final_image.device

                sun_rgb = torch.as_tensor(self.sun_state["color"], device=device)
                atten_param = torch.as_tensor(attn_param, device=device)

                sun_y = torch.empty((), device=device).uniform_(H * top_y, H * bottom_y)
                sun_pos = torch.tensor([W * 0.5, sun_y], device=device)

                x = torch.arange(W, device=device) + 0.5
                y = torch.arange(H, device=device) + 0.5
                grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
                dx = grid_x - sun_pos[0]
                dy = grid_y - sun_pos[1]
                dist = torch.sqrt(dx * dx + dy * dy) / math.sqrt(H * H + W * W)
                if attn_param.sum() == 0 and self.aug_severity == 0:
                    alpha = torch.zeros_like(attn_param)
                else:
                    alpha = -torch.log(
                        torch.tensor(max_influence, device=device) / (sun_rgb)
                    ) / (atten_param * dist.min())

                sun_img = (
                    sun_rgb[None, None, :]
                    * torch.exp(-dist[..., None] * atten_param * alpha)
                ).clamp(0.0, 1.0)

                if self.settings.enable_halo:
                    final_image = ((final_image / 255.0) + sun_img * self.aug_severity).clamp(0, 1) * 255
                    final_image2 = ((final_image2 / 255.0) + sun_img * self.aug_severity).clamp(0, 1) * 255
                    final_image = final_image.to(rgb_.device).type(torch.uint8)
                    final_image2 = final_image2.to(rgb_.device).type(torch.uint8)

        if return_attn_map:
            return (final_image, final_image2), (attenuation_map, attenuation_map2)
        return final_image, final_image2

    def apply_particles(
        self,
        rgb_: torch.Tensor,
        rgb2_: torch.Tensor | None = None,
    ):
        """
        Blend two independent particle patches onto stereo frames.

        * RNG side-effects (texture choice, crop window, opacity) always happen first
        so ablation settings share identical PRNG state.
        * Each frame gets its own random crop (could be from the same or a different
        texture).
        * Flake colour is nudged toward the current back-scatter / veiling colour
        so particles reside *inside* the haze.
        """
        H_img, W_img = rgb_.shape[:2]

        # ---------- pick ONE particle texture (must be RGBA) ---------------------
        tex_np = random.choice(self._particles_bank)  # H×W×4 uint8 expected
        if tex_np.shape[-1] != 4:
            raise RuntimeError("Particle textures should have 4 channels (RGBA).")

        H_tex, W_tex = tex_np.shape[:2]

        # ---------- helper: grab a crop (or resize) ------------------------------
        def crop_patch() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal tex_np
            if H_tex >= H_img and W_tex >= W_img:
                y0 = np.random.randint(0, H_tex - H_img + 1)
                x0 = np.random.randint(0, W_tex - W_img + 1)
                patch = tex_np[y0 : y0 + H_img, x0 : x0 + W_img]
            else:
                patch = cv2.resize(
                    tex_np, (W_img, H_img), interpolation=cv2.INTER_LINEAR
                )

            patch = torch.from_numpy(patch).to(rgb_.device).float() / 255.0  # → [0,1]
            rgb_patch = patch[..., :3]  # H×W×3
            alpha_patch = patch[..., 3]  # H×W
            return rgb_patch, alpha_patch

        rgb_left, alpha_left = crop_patch()
        rgb_right, alpha_right = (
            crop_patch() if rgb2_ is not None else (rgb_left, alpha_left)
        )

        opacity_mult = 1  # decided this was redundant
        alpha_left = alpha_left * opacity_mult * self.aug_severity
        alpha_right = alpha_right * opacity_mult * self.aug_severity
        alpha_left = alpha_left.unsqueeze(-1)  # H×W×1
        alpha_right = alpha_right.unsqueeze(-1)

        bcol = self._current_backscatter_color.to(rgb_left.device)  # (3,)
        tint_strength = 0.6
        rgb_left = rgb_left * (1.0 - tint_strength) + bcol * tint_strength
        rgb_right = rgb_right * (1.0 - tint_strength) + bcol * tint_strength

        if not self.settings.enable_particles:
            if not self._printed_particles:
                print("NOT applying particles")
                self._printed_particles = True
            return rgb_, rgb2_

        if not self._printed_particles:
            print("Applying particles")
            self._printed_particles = True

        def blend(base_img: torch.Tensor, rgb_tex: torch.Tensor, alpha: torch.Tensor):
            base = base_img.float() / 255.0
            out = (1.0 - alpha) * base + alpha * rgb_tex
            return (out * 255.0).clamp(0, 255).to(torch.uint8)

        out_left = blend(rgb_, rgb_left, alpha_left)
        out_right = blend(rgb2_, rgb_right, alpha_right) if rgb2_ is not None else None
        return out_left, out_right
