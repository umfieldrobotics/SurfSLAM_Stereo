import torch
import torch.nn.functional as F
import warnings
import kornia
from math import ceil

warnings.filterwarnings("ignore")


def rgb_to_grayscale(img: torch.Tensor) -> torch.Tensor:
    """
    Convert a CUDA RGB tensor of shape ([B?], 3, H, W) to grayscale.
    """
    in_dim = img.dim()
    if in_dim == 3:
        img = img[None]

    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError("Input tensor must have shape ([B?], 3, H, W)")

    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    if in_dim == 3:
        return gray  # Return shape (1,H,W)
    return gray[:, None]  # Return shape (B, 1, H, W)


def simulate_light(
    light_position_: torch.Tensor,
    light_power_: float,
    light_color_: torch.Tensor,
    rgb_: torch.Tensor,
    normals_: torch.Tensor,
    pts_: torch.Tensor,
    ambient_light_: float = 0.0,
    is_sun: bool = False,
    sun_height: float = 0.5,
    sun_dir: torch.Tensor = None,
    aug_severity: float = 1.0,
) -> torch.Tensor:
    """
    Ideally everything that we can get from the dataloader

    Parameters:
        light_position  : The position of the light source in world coordinates. (3,)
        light_power     : The power of the light source. (scalar)
        rgb             : The RGB image. (H, W, 3)
        depth           : The depth image. (H, W)
        normals         : The normals of the surface. (H, W, 3)
        pose            : The pose of the camera in the world frame. (4, 4)
        cam_intrinsics  : The camera intrinsics matrix. (3, 3)
        ambient_light   : The ambient light factor. (scalar)
    Returns:
        lit_image    : The lit image. (H, W, 3)
    """
    points = pts_.permute(1, 2, 0)
    light_col = light_color_.to(rgb_.device)
    light_position = light_position_.to(rgb_.device)

    # compute the normals in the camera coordinate frame
    if is_sun:
        if sun_dir is None:
            light_dir = torch.tensor([0.0, 1.0, 1.0], device=rgb_.device)
        else:
            light_dir = sun_dir.to(rgb_.device)

        distance = (points[..., 1:2] - points[..., 1:2].min()) + sun_height
    else:
        light_dir = light_position - points
        distance = torch.norm(light_dir, dim=-1, keepdim=True)

    light_intensity_dir = light_dir / (distance + 1e-6)

    # negating since the normals computed in the dataloader are facing outward
    # point light attenuation model (https://www.cemyuksel.com/research/pointlightattenuation/pointlightattenuation.pdf)
    # assuming the point to be small enough that we approximate the attenuation as the inverse distance
    cos_theta = (
        (normals_ * -light_intensity_dir).sum(dim=-1, keepdim=True).clamp(min=0.0)
    )
    attn = 1.0 / (distance**2 + 1e-6)
    diffuse = light_col * light_power_ * cos_theta * attn
    ambient = ambient_light_ * light_col
    shading = (diffuse + ambient).clamp(0.0, 1.0) * aug_severity

    lit_image = (rgb_ / 255 * (1 + shading)).clamp(0.0, 1.0)
    lit_image = (lit_image * 255).to(torch.uint8)
    return lit_image


def perturb_normals_toward_direction(
    normals: torch.Tensor,
    target_direction: torch.Tensor,
    max_perturbation: float = 1.0,
    random_seed: int = None,
) -> torch.Tensor:
    """
    Perturb each normal in `normals` toward `target_direction` by a random factor
    up to `max_perturbation`.

    Args:
        normals (torch.Tensor): (H, W, 3) array of unit normals.
        target_direction (torch.Tensor): (3,) target direction (will be normalized).
        max_perturbation (float): Maximum fraction to blend toward the target direction.
            - 0 means no change,
            - 1 means some pixels can be fully replaced by target_direction,
            - any value in between scales that effect.
        random_seed (int, optional): Set for reproducible random results.

    Returns:
        torch.Tensor: (H, W, 3) array of new perturbed normals (all normalized).
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    h, w, _ = normals.shape

    # Ensure the target direction is normalized
    t_dir = target_direction / (torch.norm(target_direction) + 1e-8)

    # Generate a random factor in [0, max_perturbation] for each pixel
    # shape: (H, W)
    r_factors = torch.ones((h, w)).to(normals.device) * max_perturbation

    # Expand to (H, W, 1) for broadcasting
    r_factors_3d = r_factors[..., None]

    # Weighted combination of original normal + target direction
    # new_normal = normalize( (1 - r) * N_orig + r * T )
    # We'll compute each pixel's new normal and then normalize.
    # For speed, we can do it in a vectorized way:
    new_normals = (1.0 - r_factors_3d) * normals + r_factors_3d * t_dir

    # Normalize
    lengths = torch.norm(new_normals, dim=-1, keepdim=True) + 1e-8
    new_normals = new_normals / lengths

    return new_normals


def tiled_aspect_crops_upscaled(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args
    ----
    img : (C, H, W) CUDA tensor of any dtype (uint8, float16/32, …)

    Returns
    -------
    two (C, H, W) CUDA tensors:
        • left-aligned and right-aligned crops
        • aspect ratio preserved, tiled by stride (2*W_c - s = W)
        • each up-scaled to original resolution
    """
    C, H, W = img.shape
    device = img.device
    orig_dtype = img.dtype

    W_c_min = (W // 2) + 1
    W_c_max = W - 1
    W_c = torch.randint(W_c_min, W_c_max + 1, (), device=device).item()
    H_c = ceil(W_c * H / W)

    stride = W - W_c

    # random vertical start so crop fits
    y0_max = H - H_c
    y0 = torch.randint(0, y0_max + 1, (), device=device).item()
    y_slice = slice(y0, y0 + H_c)

    # ── grab the two crops (pure slicing) ──────────────────────────────────────
    crop_L = img[:, y_slice, 0:W_c]
    crop_R = img[:, y_slice, stride : stride + W_c]

    # ── helper: upscale to full res, preserving dtype ─────────────────────────
    def upscale(crop: torch.Tensor) -> torch.Tensor:
        x = crop.to(torch.float32)  # float for op
        x = F.interpolate(
            x.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
        return (
            x.round().clamp(0, 255).to(orig_dtype)  # back to orig
            if not orig_dtype.is_floating_point
            else x.to(orig_dtype)
        )

    return upscale(crop_L), upscale(crop_R)


def blend_temporal_caustics_with_geometry(
    rgb_: torch.Tensor,
    depth_: torch.Tensor,
    normals_: torch.Tensor,
    rgb2_: torch.Tensor,
    depth2_: torch.Tensor,
    normals2_: torch.Tensor,
    caustic_texture_: torch.Tensor,
    sun_dir_: torch.Tensor = None,
    blend_weight_=1.0,
    blend_spread_: float = 0.0,
    normal_perturbation_: float = None,
    aug_severity: float = 1.0,
):
    """
    Simulate caustics using a texture image (i.e. a .png)

    Parameters:
        rgb             : HxWx3 RGB image.
        depth           : HxW depth image (per-pixel depth values).
        normals         : HxWx3 array of per-pixel normals (in the camera coordinate frame, assumed normalized).
        caustic_texture : Caustic texture image (grayscale or color). If it doesn't match the
                        rgb resolution, it will be resized.
        light_dir_        : The direction that the caustics will be amplified the most in, in the world frame
        blend_weight    : Base blending weight (0 to 1) controlling the maximum contribution of the caustics.

    Returns:
        result          : The RGB image with the caustic texture blended in.
    """
    h, w = depth_.shape

    if sun_dir_ is None:
        sun_dir_ = torch.tensor([0.0, 1.0, 1.0]).to(rgb_.device)

    # Ensure image is in CHW format
    is_color = caustic_texture_.ndim == 3
    if is_color:
        caustic_texture_ = caustic_texture_.permute(2, 0, 1)  # HWC -> CHW
    else:
        caustic_texture_ = caustic_texture_.unsqueeze(0)  # HW -> CHW

    # Resize if needed
    caustic_texture1, caustic_texture2 = tiled_aspect_crops_upscaled(caustic_texture_)
    if caustic_texture1.shape[1:] != (h, w):
        caustic_texture1 = kornia.geometry.resize(
            caustic_texture1.unsqueeze(0),
            (h, w),
            interpolation="bilinear",
            align_corners=False,
        ).squeeze(0)
        caustic_texture2 = kornia.geometry.resize(
            caustic_texture2.unsqueeze(0),
            (h, w),
            interpolation="bilinear",
            align_corners=False,
        ).squeeze(0)
    # Convert BGR to grayscale
    if is_color:
        caustic_texture1 = rgb_to_grayscale(caustic_texture1)
        caustic_texture2 = rgb_to_grayscale(caustic_texture2)

    caustic_texture1 = caustic_texture1.squeeze(0)
    caustic_texture1 = caustic_texture1[:, :, None]
    caustic_texture2 = caustic_texture2.squeeze(0)
    caustic_texture2 = caustic_texture2[:, :, None]

    sun_dir_ /= torch.norm(sun_dir_)

    if normal_perturbation_ is not None:
        normals_ = perturb_normals_toward_direction(
            normals=normals_,
            target_direction=sun_dir_,
            max_perturbation=normal_perturbation_,
            random_seed=None,
        )
        normals2_ = perturb_normals_toward_direction(
            normals=normals2_,
            target_direction=sun_dir_,
            max_perturbation=normal_perturbation_,
            random_seed=None,
        )

    dot_normals1 = torch.clip(torch.sum(normals_ * sun_dir_, dim=2), 0, 1)
    dot_normals2 = torch.clip(torch.sum(normals2_ * sun_dir_, dim=2), 0, 1)

    # # sample the blend factor based on normals and depth
    # if blend_spread_ > 0:
    #     # perturb the bend_weight by blend_spread_ amount as a scalar value only
    #     blend_weight_ = torch.randn(1).to("cuda") * blend_spread_**2 + blend_weight_
    # blend_weight_ = torch.clip(blend_weight_, 0, 1)

    blend_factor1 = blend_weight_ * dot_normals1 * aug_severity  # * depth_weight
    blend_factor1_3 = torch.repeat_interleave(blend_factor1[:, :, None], 3, dim=2)
    blend_factor2 = blend_weight_ * dot_normals2 * aug_severity  # * depth_weight
    blend_factor2_3 = torch.repeat_interleave(blend_factor2[:, :, None], 3, dim=2)

    result = (
        rgb_.float() * (1 - blend_factor1_3)
        + caustic_texture1.float() * blend_factor1_3
    ).float()
    result2 = (
        rgb2_.float() * (1 - blend_factor2_3)
        + caustic_texture2.float() * blend_factor2_3
    ).float()

    result = torch.clip(result, 0, 255).to(torch.uint8)
    result2 = torch.clip(result2, 0, 255).to(torch.uint8)

    return result, result2


def blend_caustics_with_geometry(
    rgb_: torch.Tensor,
    depth_: torch.Tensor,
    normals_: torch.Tensor,
    caustic_texture_: torch.Tensor,
    sun_dir_: torch.Tensor = None,
    blend_weight_=1.0,
    blend_spread_: float = 0.0,
    normal_perturbation_: float = None,
):
    """
    Simulate caustics using a texture image (i.e. a .png)

    Parameters:
        rgb             : HxWx3 RGB image.
        depth           : HxW depth image (per-pixel depth values).
        normals         : HxWx3 array of per-pixel normals (in the camera coordinate frame, assumed normalized).
        caustic_texture : Caustic texture image (grayscale or color). If it doesn't match the
                        rgb resolution, it will be resized.
        light_dir_        : The direction that the caustics will be amplified the most in, in the world frame
        blend_weight    : Base blending weight (0 to 1) controlling the maximum contribution of the caustics.

    Returns:
        result          : The RGB image with the caustic texture blended in.
    """
    h, w = depth_.shape

    if sun_dir_ is None:
        sun_dir_ = torch.tensor([0.0, 1.0, 1.0]).to(rgb_.device)

    # Ensure image is in CHW format
    is_color = caustic_texture_.ndim == 3
    if is_color:
        caustic_texture_ = caustic_texture_.permute(2, 0, 1)  # HWC -> CHW
    else:
        caustic_texture_ = caustic_texture_.unsqueeze(0)  # HW -> CHW

    # Resize if needed
    if caustic_texture_.shape[1:] != (h, w):
        caustic_texture_ = kornia.geometry.resize(
            caustic_texture_.unsqueeze(0),
            (h, w),
            interpolation="bilinear",
            align_corners=False,
        ).squeeze(0)

    # Convert BGR to grayscale
    if is_color:
        caustic_texture_ = kornia.color.bgr_to_grayscale(caustic_texture_)

    caustic_texture_ = caustic_texture_.squeeze(0)
    caustic_texture_ = caustic_texture_[:, :, None]

    sun_dir_ /= torch.norm(sun_dir_)

    if normal_perturbation_ is not None:
        normals_ = perturb_normals_toward_direction(
            normals=normals_,
            target_direction=sun_dir_,
            max_perturbation=normal_perturbation_,
            random_seed=None,
        )

    dot_normals = torch.clip(torch.sum(normals_ * sun_dir_, dim=2), 0, 1)

    blend_factor = blend_weight_ * dot_normals  # * depth_weight
    blend_factor_3 = torch.repeat_interleave(blend_factor[:, :, None], 3, dim=2)

    result = (
        rgb_.float() * (1 - blend_factor_3) + caustic_texture_.float() * blend_factor_3
    ).float()

    result = torch.clip(result, 0, 255).to(torch.uint8)

    return result


def attenuate_light_source(
    light_position_: torch.Tensor,
    light_color_: torch.Tensor,
    rgb_: torch.Tensor,
    points_: torch.Tensor,
    attn_params_: torch.Tensor,
    bs_params_: torch.Tensor,
    veiling_light_params_: torch.Tensor,
):
    """
    Simulate a light source at ```light_position``` and attenuate the ```light_color```
    based on the distance from the light source to the surface.

    Parameters:
        light_position_         : The position of the light source in world coordinates. (3,)
        light_color_            : The color of the light source. (3,)
        rgb_                    : The RGB image. (H, W, 3)
        depth_                  : The depth image. (H, W)
        pose_                   : The pose of the camera in the world frame. (4, 4)
        cam_intrinsics_         : The camera intrinsics matrix. (3, 3)
        attn_params_            : Attenuation parameters for the light source. (scalar)
        bs_params_              : Background scattering parameters for the light source. (scalar)
        veiling_light_params_   : Veiling light parameters for the light source. (scalar)
    Returns:
        final_image             : The final image after applying attenuation the light source. (H, W, 3)
    """

    light_dir = light_position_.to(rgb_.device).view(1, 1, 3) - points_.permute(
        1, 2, 0
    )  # (3, H, W)
    distance = torch.norm(light_dir, dim=-1, keepdim=True)

    attn_component = light_color_.to(rgb_.device) * torch.exp(-distance * attn_params_)
    bs_component = veiling_light_params_ * (1 - torch.exp(-distance * bs_params_))

    final_image = (attn_component + bs_component).clamp(0, 1)  # (H, W, 3)
    return final_image
