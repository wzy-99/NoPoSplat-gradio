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
    from src.model.ply_export import export_ply


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


class APP:
    images: np.ndarray

    def __init__(self, model, output_dir, resolution=256):
        self.model = model
        self.output_dir = output_dir
        if type(resolution) == int:
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution

    def launch(self):
        with gr.Blocks() as app:
            with gr.Row():
                with gr.Column():
                    images = gr.Gallery(label="Input", type="numpy")
                    fx = gr.Number(label="fx", value=0.8, minimum=0.0, maximum=2.0, step=0.01)
                    fy = gr.Number(label="fy", value=0.8, minimum=0.0, maximum=2.0, step=0.01)
                    cx = gr.Number(label="cx", value=0.5, minimum=0.0, maximum=1.0, step=0.01)
                    cy = gr.Number(label="cy", value=0.5, minimum=0.0, maximum=1.0, step=0.01)
                    auto = gr.Checkbox(label="Auto-adjust", value=True)
                    apply = gr.Button(value="Apply")
                with gr.Column():
                    output = gr.Model3D(label="Output")
            
            images.upload(self.on_upload, images)
            apply.click(self.on_apply, [cx, cy, fx, fy, auto], output)

        app.launch()

    def on_upload(self, inps: np.ndarray):
        images = []
        for (image, path) in inps:
            h, w = image.shape[:2]

            if image.shape[2] == 4: # Remove alpha channel if present.
                image = image[:, :, :3]

            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            
            cropped_image = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
            images.append(cropped_image)

        self.images = np.stack(images)

    def on_apply(self, cx: float, cy: float, fx: float, fy: float, auto: bool):
        images = torch.from_numpy(self.images).permute(0, 3, 1, 2).float().cuda() / 255.0
        images = torch.nn.functional.interpolate(images, size=self.resolution, mode='bilinear', align_corners=False)
        images = images.unsqueeze(0)

        if auto:
            cx, cy = 0.5, 0.5
            fx = fy = 0.8

        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().reshape(1, 1, 3, 3).repeat(1, 2, 1, 1).cuda()

        batch = BatchedExample(
            context=BatchedViews(
                intrinsics=intrinsics,
                image=images,
            )
        )
        
        with torch.no_grad() and torch.autocast(device_type="cuda"):
            gaussian, visualization_dump = self.model.predict(batch)

        paths = []
        B = 1
        for b in range(B):
            means = gaussian.means[b]
            covariances = gaussian.covariances[b]
            harmonics = gaussian.harmonics[b]
            opacities = gaussian.opacities[b]
            rotations = visualization_dump['rotations'][b]
            scales = visualization_dump['scales'][b]
            export_ply(
                extrinsics=None,
                means=means,
                scales=scales,
                rotations=rotations,
                harmonics=harmonics,
                opacities=opacities,
                path=Path(self.output_dir) / f"output_{b}.ply",
            )
            paths.append(os.path.join(self.output_dir, f"output_{b}.ply"))

        return paths[-1] # TODO: return all paths


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

    app = APP(model_wrapper, output_dir=output_dir, resolution=cfg_dict.resolution)

    app.launch()


if __name__ == "__main__":
    train()
