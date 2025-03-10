import json
import os
from pathlib import Path

import torch.nn as nn
import torch
from typing import Callable, List, Tuple
from model.utils.extraction import instantiate_extractor
from model.vitac.vtt_reall import VTT_ReAll


import torchvision.transforms as T
from torchvision.models import get_model, list_models, ResNet

from einops import rearrange
import numpy as np
DEFAULT_CACHE = "cache/"
NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class Encoder(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim, en_mode='cls'):
        super(Encoder, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.pretrain_dir = pretrain_dir
        self.model_name = model_name
        self.en_mode = en_mode
        self.backbone, self.preprocess, gap_dim = load(model_id=model_name, freeze=freeze, cache=pretrain_dir)
        if self.en_mode != 'cls':
            self.projector = nn.Sequential(
                instantiate_extractor(self.backbone, n_latents=1)(),
                nn.Linear(gap_dim, emb_dim))
        else:
            self.projector = nn.Linear(gap_dim, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        self.freeze = freeze


    @torch.no_grad()
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = self.preprocess(x)
            feat = self.backbone.get_representations(x, mode=self.en_mode)
        elif isinstance(x, tuple):
            imag, tac = x
            imag = self.preprocess(imag)
            feat = self.backbone.get_representations(imag, tac, mode=self.en_mode)
        else:
            raise AttributeError(f'type of input {x.type} is not expected')
        return self.projector(feat), feat

    def forward_feat(self, feat):
        return self.projector(feat)

    def visual_backbone(self, x):
        loss, recons, _ = self.backbone(x)
        recon_imgs = self.backbone.generate_origin_img(recons.cpu(), x.cpu())
        save_recons_imgs(x.cpu(), recon_imgs.cpu(),
                         Path(self.pretrain_dir),
                         identify=f"train_{self.model_name}",
                         online_normalization=NORMALIZATION)

class Encoder2(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim, en_mode='cls'):
        super(Encoder2, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.pretrain_dir = pretrain_dir
        assert isinstance(model_name, list)
        self.model_name = model_name
        self.en_mode = en_mode
        self.backbone_img, self.preprocess, gap_dim_img = load(model_id=model_name[0], freeze=freeze, cache=pretrain_dir)
        self.backbone_tac, _, gap_dim_tac = load(model_id=model_name[1], freeze=freeze, cache=pretrain_dir)
        if self.en_mode != 'cls':
            self.projector_img = nn.Sequential(
                instantiate_extractor(self.backbone_img, n_latents=1)(),
                nn.Linear(gap_dim_img, emb_dim))
            self.projector_tac = nn.Sequential(
                instantiate_extractor(self.backbone_tac, n_latents=1)(),
                nn.Linear(gap_dim_tac, emb_dim))
        else:
            self.projector_img = nn.Linear(gap_dim_img, emb_dim)
            self.projector_tac = nn.Linear(gap_dim_tac, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        self.emb_dim = emb_dim
        self.freeze = freeze


    @torch.no_grad()
    def forward(self, x):
        # if isinstance(x, torch.Tensor):
        #     x = self.preprocess(x)
        #     feat = self.backbone.get_representations(x, mode=self.en_mode)
        if isinstance(x, tuple):
            imag, tac = x
            imag = self.preprocess(imag)
            feat_img = self.backbone_img.get_representations(imag, mode=self.en_mode)
            p_feat_img = self.projector_img(feat_img)
            feat_tac = self.backbone_tac.get_representations(tac, mode=self.en_mode)
            p_feat_tac = self.projector_tac(feat_tac)
        else:
            raise AttributeError(f'type of input {x.type} is not expected')

        return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat([feat_img, feat_tac], dim=1)

    def forward_feat(self, feat):
        feat_img, feat_tac = feat[:, :-20, :], feat[:, -20:, :]
        p_feat_img = self.projector_img(feat_img)
        p_feat_tac = self.projector_tac(feat_tac)
        return torch.cat([p_feat_img, p_feat_tac], dim=-1)

    def visual_backbone(self, x):
        loss, recons, _ = self.backbone(x)
        recon_imgs = self.backbone.generate_origin_img(recons.cpu(), x.cpu())
        save_recons_imgs(x.cpu(), recon_imgs.cpu(),
                         Path(self.pretrain_dir),
                         identify=f"train_{self.model_name[0]}",
                         online_normalization=NORMALIZATION)
class Encoder_T(nn.Module):
    def __init__(self, model_name, pretrain_dir, freeze, emb_dim, en_mode='cls'):
        super(Encoder_T, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.pretrain_dir = pretrain_dir
        self.model_name = model_name
        self.en_mode = en_mode
        self.backbone, _, gap_dim = load(model_id=model_name, freeze=freeze, cache=pretrain_dir)
        if self.en_mode != 'cls':
            self.projector = nn.Sequential(
                instantiate_extractor(self.backbone, n_latents=1)(),
                nn.Linear(gap_dim, emb_dim))
        else:
            self.projector = nn.Linear(gap_dim, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        self.freeze = freeze


    @torch.no_grad()
    def forward(self, x):

        feat = self.backbone.get_representations(x, mode=self.en_mode)

        return self.projector(feat), feat

    def forward_feat(self, feat):
        return self.projector(feat)

class Encoder_no_pre(nn.Module):

    def __init__(self, model_name, emb_dim):
        super(Encoder_no_pre, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.model_name = model_name
        # self.en_mode = en_mode
        assert self.model_name in list_models(), f"{self.model_name} is not included in {list_models()}"

        self.backbone = get_model(model_name) # using no weights
        gap_dim = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) # abandon the last two layers

        self.preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ])

        self.projector_img = nn.Linear(gap_dim, emb_dim)
        self.projector_tac = nn.Linear(20, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        # self.freeze = freeze
    def forward(self, x):

        if isinstance(x, torch.Tensor):
            imag, tac = x[:, :-20], x[:, -20:]
            imag = imag.view(-1, 224, 224, 3).permute(0, 3, 1, 2).to(torch.uint8)  # image
        elif isinstance(x, tuple):
            imag, tac = x
        else:
            raise AssertionError
        imag = self.preprocess(imag)
        img_feat = self.backbone(imag)
        img_feat = torch.flatten(img_feat, 1)
        img_feat = self.projector_img(img_feat)
        tac_feat = self.projector_tac(tac)

        feat = torch.cat([img_feat, tac_feat], dim=-1)


        return feat
class Encoder_CNN(nn.Module):

    def __init__(self, model_name, emb_dim):
        super(Encoder_CNN, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.model_name = model_name
        # self.en_mode = en_mode
        assert self.model_name =='CNN', f"{self.model_name} is not CNN"
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ])

        self.projector_img = nn.Linear(64*56*56, emb_dim)
        self.projector_tac = nn.Linear(20, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        # self.freeze = freeze
    def forward(self, x):

        if isinstance(x, torch.Tensor):
            imag, tac = x[:, :-20], x[:, -20:]
            imag = imag.view(-1, 224, 224, 3).permute(0, 3, 1, 2).to(torch.uint8)  # image
        elif isinstance(x, tuple):
            imag, tac = x
        else:
            raise AssertionError
        imag = self.preprocess(imag)
        img_feat = self.relu(self.conv1(imag))
        img_feat = self.maxpool(img_feat)
        img_feat = self.relu(self.conv2(img_feat))
        img_feat = self.maxpool(img_feat)
        img_feat = img_feat.contiguous().view(img_feat.size(0), -1)

        img_feat = self.projector_img(img_feat)
        tac_feat = self.projector_tac(tac)

        feat = torch.cat([img_feat, tac_feat], dim=-1)


        return feat


MODEL_REGISTRY = {
    "vt20t-reall-tmr05-bin-ft+dataset-BottleCap": {
        "config": "model/vitac/model_and_config/vt20t-reall-tmr05-bin-ft+dataset-BottleCap.json",
        "checkpoint": "model/vitac/model_and_config/vt20t-reall-tmr05-bin-ft+dataset-BottleCap.pt",
        "cls": VTT_ReAll,
    }
    # if youa want to add extra pretrained models, add model path here

}
def load(model_id: str, freeze: bool = True, cache: str = DEFAULT_CACHE, device: torch.device = "cpu"):
    """
    Download & cache specified model configuration & checkpoint, then load & return module & image processor.

    Note :: We *override* the default `forward()` method of each of the respective model classes with the
            `extract_features` method --> by default passing "NULL" language for any language-conditioned models.
            This can be overridden either by passing in language (as a `str) or by invoking the corresponding methods.
    """
    assert model_id in MODEL_REGISTRY, f"Model ID `{model_id}` not valid, try one of  {list(MODEL_REGISTRY.keys())}"

    # Download Config & Checkpoint (if not in cache)
    # model_cache = Path(cache) / model_id
    config_path, checkpoint_path = Path(cache) / f"{model_id}.json", Path(cache) / f"{model_id}.pt"
    # os.makedirs(model_cache, exist_ok=True)
    assert checkpoint_path.exists() and config_path.exists(), f'{checkpoint_path} or {config_path} model path does not exist'
    # if not checkpoint_path.exists() or not config_path.exists():
    #     gdown.download(id=MODEL_REGISTRY[model_id]["config"], output=str(config_path), quiet=False)
    #     gdown.download(id=MODEL_REGISTRY[model_id]["checkpoint"], output=str(checkpoint_path), quiet=False)

    # Load Configuration --> patch `hf_cache` key if present (don't download to random locations on filesystem)
    with open(config_path, "r") as f:
        model_kwargs = json.load(f)
        # if "hf_cache" in model_kwargs:
        #     model_kwargs["hf_cache"] = str(Path(cache) / "hf-cache")

    # By default, the model's `__call__` method defaults to `forward` --> for downstream applications, override!
    #   > Switch `__call__` to `get_representations`
    # MODEL_REGISTRY[model_id]["cls"].__call__ = MODEL_REGISTRY[model_id]["cls"].get_representations

    # Materialize Model (load weights from checkpoint; note that unused element `_` are the optimizer states...)
    model = MODEL_REGISTRY[model_id]["cls"](**model_kwargs)
    if model_id in ['VMVP']:
        state_dict, _ = torch.load(checkpoint_path, map_location=device)
        emb_dim = model_kwargs['encoder_embed_dim']
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']
        emb_dim = model_kwargs['encoder_decoder_cfg']['encoder_embed_dim']
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Freeze model parameters if specified (default: True)
    if freeze:
        for _, param in model.named_parameters():
            param.requires_grad = False

    # Build Visual Preprocessing Transform (assumes image is read into a torch.Tensor, but can be adapted)
    if model_id in list(MODEL_REGISTRY.keys()):
        # All models except R3M are by default normalized subject to default IN1K normalization...
        preprocess = T.Compose(
            [
                # T.Resize(model_kwargs["dataset_cfg"]["resolution"]),
                # T.CenterCrop(model_kwargs["resolution"]),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ]
        )
    else:
        raise AttributeError(F'{model_id} dose not exit')

    return model, preprocess, emb_dim
    # return model, emb_dim

def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())

def save_recons_imgs(
    ori_imgs: torch.Tensor,
    recons_imgs: torch.Tensor,
    save_dir_input: Path,
    identify: str,
    online_normalization
) -> None:
    # import cv2
    import cv2
    # de online transforms function
    def de_online_transform(ori_img, recon_img, norm=online_normalization):
        # rearrange
        ori_imgs = rearrange(ori_img,"c h w -> h w c")
        recon_imgs = rearrange(recon_img, "c h w -> h w c")
        # to Numpy format
        ori_imgs = ori_imgs.detach().numpy()
        recon_imgs = recon_imgs.detach().numpy()
        # DeNormalize
        ori_imgs = np.array(norm[0]) + ori_imgs * np.array(norm[1])
        recon_imgs = np.array(norm[0]) + recon_imgs * np.array(norm[1])
        # to cv format
        ori_imgs = np.uint8(ori_imgs * 255)
        recon_imgs = np.uint8(recon_imgs * 255)

        return ori_imgs, recon_imgs

    save_dir = save_dir_input / identify
    os.makedirs(str(save_dir), exist_ok=True)
    for bid in range(ori_imgs.shape[0]):
        ori_img = ori_imgs[bid]
        recon_img = recons_imgs[bid]
        # de online norm
        ori_img, recon_img = de_online_transform(ori_img, recon_img)

        ori_save_path = save_dir / f"{bid}_raw.jpg"
        recon_save_path = save_dir / f"{bid}_recon.jpg"

        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(ori_save_path), ori_img)
        cv2.imwrite(str(recon_save_path), recon_img)