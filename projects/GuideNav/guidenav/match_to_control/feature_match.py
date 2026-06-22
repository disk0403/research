import os
import numpy as np
import random
import cv2
from math import atan2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
import time

# Get the directory containing this file for relative imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.join(_CURRENT_DIR, 'methods')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _prepend_paths(paths):
    for path in reversed(paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _clear_external_models_namespace():
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.'):
            del sys.modules[name]


def _prepare_loftr():
    _prepend_paths([
        _METHODS_DIR,
        os.path.join(_METHODS_DIR, 'LoFTR'),
    ])


def _prepare_liftfeat():
    _clear_external_models_namespace()
    _prepend_paths([
        os.path.join(_METHODS_DIR, 'LiftFeat'),
        _METHODS_DIR,
    ])


def _prepare_mast3r():
    _clear_external_models_namespace()
    _prepend_paths([
        os.path.join(_METHODS_DIR, 'mast3r', 'dust3r', 'croco'),
        os.path.join(_METHODS_DIR, 'mast3r', 'dust3r'),
        os.path.join(_METHODS_DIR, 'mast3r'),
        _METHODS_DIR,
    ])


def _prepare_reloc3r(clear_models=True):
    if clear_models:
        _clear_external_models_namespace()
    _prepend_paths([
        os.path.join(_METHODS_DIR, 'reloc3r', 'croco'),
        os.path.join(_METHODS_DIR, 'reloc3r'),
        _METHODS_DIR,
    ])

def read_depth_image(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # Preserve depth precision
    if depth is None:
        raise ValueError(f"Cannot read depth image {path}.")
    return depth

def init_loftr():
    _prepare_loftr()
    from LoFTR.src.loftr import LoFTR, default_cfg

    matcher = LoFTR(config=default_cfg)
    loftr_weights = os.path.join(_METHODS_DIR, 'LoFTR', 'weights', 'outdoor_ds.ckpt')
    matcher.load_state_dict(torch.load(loftr_weights)['state_dict'])
    matcher = matcher.eval().to(device=device)
    return matcher
    
def matching_features_loftr(img1, img2, matcher):
    
    # Preprocess images
    # img1_tensor = loftr_utils_preprocess(img1, resize=(640, 480), device=device)
    # img2_tensor = loftr_utils_preprocess(img2, resize=(640, 480), device=device)
    img1_tensor = loftr_utils_preprocess(img1, resize=(640, 360), device=device)
    img2_tensor = loftr_utils_preprocess(img2, resize=(640, 360), device=device)
    
    batch = {'image0': img1_tensor, 'image1': img2_tensor}
    with torch.no_grad():
        matcher(batch)
        
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    
    kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in mkpts0]
    kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in mkpts1]
    
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(mkpts0))]

    return kp1, kp2, matches

def process_resize(w, h, resize):
    if len(resize) == 2:
        return resize
    elif len(resize) == 1 and resize[0] > 0:
        max_dim = resize[0]
        scale = max_dim / max(w, h)
        return int(w * scale), int(h * scale)
    elif len(resize) == 1 and resize[0] == -1:
        return w, h  
    else:
        raise ValueError("Invalid resize config")

def loftr_utils_preprocess(img: np.ndarray, resize=(640, 480), device='cuda'):
    w, h = img.shape[1], img.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(gray / 255.).float()[None, None].to(device)

def init_roma():
    from romatch import roma_outdoor

    roma_model = roma_outdoor(device=device)
    return roma_model

def matching_features_roma(img1_path, img2_path, roma_model):
    W_1, H_1 = Image.open(img1_path).size
    W_2, H_2 = Image.open(img2_path).size

    warp, certainty = roma_model.match(img1_path, img2_path)
    matches_raw, certainty = roma_model.sample(warp, certainty)
    top_k = int(0.2 * matches_raw.shape[0])  # top 20%, 50, 80
    indices = torch.topk(certainty, top_k).indices
    matches_raw = matches_raw[indices]

    # convert to pixel coord (roma generates in [-1, 1] x [-1, 1])
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches_raw, H_1, W_1, H_2, W_2)

    kpts1 = kpts1.cpu().numpy()
    kpts2 = kpts2.cpu().numpy()
    
    kp1 = [cv2.KeyPoint(float(x), float(y), size=1) for x, y in kpts1]
    kp2 = [cv2.KeyPoint(float(x), float(y), size=1) for x, y in kpts2]

    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(kp1))]
    print(f"ROMA matches: {len(matches)}")

    return kp1, kp2, matches

def init_mast3r():
    _prepare_mast3r()
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from mast3r.model import AsymmetricMASt3R

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    matcher = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    return matcher
    
def matching_features_mast3r(img1, img2, model):
    _prepare_mast3r()
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images_online

    images = [img1, img2]
    processed_images = load_images_online(images, size=512)
    output = inference([tuple(processed_images)], model, device, batch_size=1, verbose=False)

    # Extract predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    # Extract descriptors
    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()

    # Find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2,
        subsample_or_initxy1=8,
        device=device,
        dist='dot',
        block_size=2**13
    )
    
    # Get original image dimensions (true_shape contains original dimensions)
    H0, W0 = view1['true_shape'][0]
    H1, W1 = view2['true_shape'][0]
    
    # Filter out matches near borders
    border = 3
    valid_matches_im0 = (
        (matches_im0[:, 0] >= border) & (matches_im0[:, 0] < int(W0) - border) & 
        (matches_im0[:, 1] >= border) & (matches_im0[:, 1] < int(H0) - border)
    )
    
    valid_matches_im1 = (
        (matches_im1[:, 0] >= border) & (matches_im1[:, 0] < int(W1) - border) & 
        (matches_im1[:, 1] >= border) & (matches_im1[:, 1] < int(H1) - border)
    )
    
    valid_matches = valid_matches_im0 & valid_matches_im1
    final_matches_im0 = matches_im0[valid_matches]
    final_matches_im1 = matches_im1[valid_matches]
    
    # Convert to cv2.KeyPoint format
    kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in final_matches_im0]
    kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in final_matches_im1]
    
    # Create cv2.DMatch objects
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(final_matches_im0))]
    
    return kp1, kp2, matches

def init_liftFeat():
    _prepare_liftfeat()
    from LiftFeat.models.liftfeat_wrapper import LiftFeat, MODEL_PATH

    matcher=LiftFeat(weight=MODEL_PATH,detect_threshold=0.35)
    return matcher

def matching_features_liftFeat(img1, img2, model):
    mkpts0,mkpts1=model.match_liftfeat(img1,img2)
    
    # Apply RANSAC filtering
    if len(mkpts0) > 4:
        H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
        # H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 5.0, maxIters=2_000, confidence=0.95)
        if mask is not None:
            mask = mask.flatten()
            mkpts0 = mkpts0[mask.astype(bool)]
            mkpts1 = mkpts1[mask.astype(bool)]

    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts0]
    kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts1]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts0))]

    return kp1, kp2, matches

def init_reloc3r():
    _prepare_reloc3r()
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model

    img_reso = '512' # or 224
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=img_reso, device=device)
    return reloc3r_relpose, img_reso


def _load_images_reloc3r_from_arrays(images, size):
    from reloc3r.utils.image import ImgNorm, _resize_pil_image

    processed_images = []
    for idx, image in enumerate(images):
        if isinstance(image, (str, os.PathLike)):
            from reloc3r.utils.image import load_images

            return load_images([str(img) for img in images], size=int(size), verbose=False)

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Reloc3r expects HxWx3 images, got {image.shape}")

        # GuideNav camera images are OpenCV-style BGR. Reloc3r expects RGB PIL.
        image_rgb = image[:, :, ::-1].astype(np.uint8)
        pil_image = Image.fromarray(image_rgb).convert('RGB')
        width_before, height_before = pil_image.size

        if int(size) == 224:
            pil_image = _resize_pil_image(
                pil_image,
                round(int(size) * max(width_before / height_before, height_before / width_before)),
            )
        else:
            pil_image = _resize_pil_image(pil_image, int(size))

        width, height = pil_image.size
        cx, cy = width // 2, height // 2
        if int(size) == 224:
            half = min(cx, cy)
            pil_image = pil_image.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if width == height:
                halfh = 3 * halfw / 4
            pil_image = pil_image.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        processed_images.append(
            {
                'img': ImgNorm(pil_image)[None],
                'true_shape': np.int32([pil_image.size[::-1]]),
                'idx': idx,
                'instance': str(idx),
            }
        )
    return processed_images


# direct inference of relative position
def matching_features_reloc3r_inv(img1, img2, model, img_reso):
    _prepare_reloc3r(clear_models=False)
    from reloc3r.utils.image import check_images_shape_format
    from reloc3r.reloc3r_relpose import inference_relpose
    from reloc3r.utils.device import to_numpy

    images = _load_images_reloc3r_from_arrays([img1, img2], size=int(img_reso))
    images = check_images_shape_format(images, device)

    # Relative pose estimation
    batch = [images[0], images[1]]
    pose2to1 = to_numpy(inference_relpose(batch, model, device, use_amp=True)[0])

    # Normalize translation to unit scale
    pose2to1[0:3, 3] = pose2to1[0:3, 3] / np.linalg.norm(pose2to1[0:3, 3])

    # Extract relative position (camera frame to robot frame)
    x_rel = pose2to1[2, 3]   # Z translation (forward/backward)
    y_rel = -pose2to1[0, 3]  # X translation (right/left)

    # Extract yaw rotation
    yaw = np.arctan2(-pose2to1[0, 2], pose2to1[2, 2])

    return x_rel, y_rel, np.degrees(yaw)
