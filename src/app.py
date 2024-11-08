import os
from pathlib import Path

import hydra
import torch
import signal
import gradio as gr
import numpy as np
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from dacite import Config, from_dict
    

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.config import DecoderCfg, EncoderCfg
    from src.dataset.types import BatchedExample, BatchedViews


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


class APP:
    images: np.ndarray

    def __init__(self, model):
        self.model = model

    def launch(self):
        with gr.Blocks() as app:
            with gr.Row():
                with gr.Column():
                    images = gr.Gallery(label="Input", type="numpy")
                    fx = gr.Number(label="fx")
                    fy = gr.Number(label="fy")
                    cx = gr.Number(label="cx")
                    cy = gr.Number(label="cy")
                    auto = gr.Checkbox(label="Auto-adjust", value=True)
                    apply = gr.Button(value="Apply")
                with gr.Column():
                    output = gr.Image(label="Output", type="numpy")
            
            images.upload(self.on_upload, images)
            apply.click(self.on_apply, [cx, cy, fx, fy, auto], output)

        app.launch()

    def on_upload(self, inps: np.ndarray):
        images = []
        for (image, path) in inps:
            images.append(image)
        self.images = np.stack(images)

    def on_apply(self, cx: float, cy: float, fx: float, fy: float, auto: bool):
        V, H, W, C = self.images.shape
        B = 1

        if auto:
            cx, cy = W / 2, H / 2
            fx = fy = min(W, H)

        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1).cuda()
        images = torch.from_numpy(self.images).permute(0, 3, 1, 2).unsqueeze(0).float().cuda() / 255.0

        batch = BatchedExample(
            context=BatchedViews(
                intrinsics=intrinsics,
                image=images,
            )
        )
        
        output = self.model.predict(batch)

        return 


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="app",
)
def train(cfg_dict: DictConfig):
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Prepare the checkpoint for loading.
    checkpoint_path: str = cfg_dict.checkpointing.load

    torch.manual_seed(cfg_dict.seed)

    encoder_cfg: EncoderCfg = from_dict(data_class=EncoderCfg, data=OmegaConf.to_container(cfg_dict.model.encoder))

    encoder, encoder_visualizer = get_encoder(encoder_cfg)

    decoder_cfg: DecoderCfg = from_dict(data_class=DecoderCfg, data=OmegaConf.to_container(cfg_dict.model.decoder))

    decoder = get_decoder(decoder_cfg)

    model_wrapper = ModelWrapper(
        None,
        None,
        None,
        encoder,
        encoder_visualizer,
        decoder,
        None,
        None,
        None,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")['state_dict']
    model_wrapper.load_state_dict(state_dict)
    model_wrapper.cuda()

    app = APP(model_wrapper)

    app.launch()


if __name__ == "__main__":
    train()
