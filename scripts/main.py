"""Reconstruct images from anomaly to normal using SAM and Stable Diffusion Inpainting"""
import os
import numpy as np
import torch
import yaml
import cv2
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
from segment_anything import build_sam, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapter
from skimage.metrics import structural_similarity


class Reconstructer:
    STRENGTHS = [0.1, 0.2, 0.5, 0.7]
    IMAGE_SIZE = (512, 512)

    def __init__(self, save_path: str, dataset: str):
        self.save_path = Path(save_path)
        self.dataset = dataset
        self.parser, self.enable_mask, self.mask_type = self._read_data_setting()
        self.seg_predictor, self.ip_model, self.processor = self._load_models()

    def _read_data_setting(self):
        with open(f"data/{self.dataset}.yaml", "r") as stream:
            parser = yaml.load(stream, Loader=yaml.CLoader)
        return parser, parser["dataset"]["mask_enable"], parser["dataset"]["mask_type"]

    def _load_models(self):
        processor = None
        if self.enable_mask:
            if self.mask_type == "CLIPSeg":
                processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
                seg_predictor = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
            else:
                # Load SAM model
                sam = build_sam(checkpoint="models/sam_vit_h_4b8939.pth")
                sam.to('cuda')
                seg_predictor = SamAutomaticMaskGenerator(
                        sam, pred_iou_thresh=0.98, stability_score_thresh=0.92, min_mask_region_area=10000)
        else:
            seg_predictor = None

        # Load stable diffusion inpainting models
        base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae_model_path = "stabilityai/sd-vae-ft-mse"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        # load SD pipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )
        pipe = pipe.to("cuda")

        # load ip-adapter
        image_encoder_path = "models/image_encoder/"
        ip_ckpt = "models/ip-adapter_sd15.bin"
        device = "cuda"
        ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
        return seg_predictor, ip_model, processor
    
    def run_all(self):
        for strength in self.STRENGTHS:
            self.strength = strength
            self.run()

    def run(self):
        for category in self.parser["categorys"]:
            print(f'Processing {category}...')

            # Load reference image
            root_path = Path(self.parser["dataset"]["path"])
            ref_folder_path = root_path / category / self.parser["dataset"]["splite"]

            for anomaly_type in os.listdir(ref_folder_path):
                self.save_folder_path = self.save_path / f"strength_{self.strength}" / category / anomaly_type
                os.makedirs(self.save_folder_path)
                for img_path in (ref_folder_path/anomaly_type).iterdir():
                    print(img_path)
                    self.reconstruct(img_path, ref_folder_path)

    def reconstruct(self, img_path: str, ref_folder_path: str):
        # Load demo image
        image_source = Image.open(img_path).convert("RGB")
        image_source = np.asarray(image_source)

        # Image Inpainting
        image_source_pil = Image.fromarray(image_source)

        if self.enable_mask:
            if self.mask_type == "CLIPSeg":
                inputs = self.processor(text=["background"], images=[image_source], padding=True, return_tensors="pt")
                # predict
                with torch.no_grad():
                    outputs = self.seg_predictor(**inputs)
                preds = torch.sigmoid(outputs.logits).numpy()*255
                ret, image_mask = cv2.threshold(preds.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            else:
                # Run the segmentation model
                masks = self.seg_predictor.generate(image_source)
                masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                image_mask = ~masks[0]['segmentation']

            image_mask_pil = Image.fromarray(image_mask)
            image_mask_for_inpaint = image_mask_pil.resize(self.IMAGE_SIZE)
            image_mask_for_inpaint.save(self.save_folder_path / f"mask_{img_path.name}")
        else:
            image_mask_for_inpaint = Image.new("L", self.IMAGE_SIZE, (255))
        
        image_source_for_inpaint = image_source_pil.resize(self.IMAGE_SIZE)
        image_source_for_inpaint.save(self.save_folder_path / f"source_{img_path.name}")

        thresh = float("inf")
        file_names = [x for x in (ref_folder_path/"good").iterdir() if x.is_file()]
        for i in range(1):
            # Load reference image
            ref_image_path = ref_folder_path / file_names[i]
            ref_image = Image.open(ref_image_path)
            ref_image.resize(self.IMAGE_SIZE)
            print(f'Reference image: {ref_image_path}')
            image_inpainting = self.ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=200,
                seed=42, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint, strength=self.strength)[0]
            thresh_new = self.detect(image_source_for_inpaint, image_inpainting, image_mask_for_inpaint)
            print('thresh', thresh_new)
            if thresh_new < thresh:
                thresh = thresh_new
                image_inpainting.save(self.save_folder_path / f"inpaint_{img_path.name}")

    def detect(self, origin_img, reconstruc_img, mask):
        before = np.array(origin_img)
        after = np.array(reconstruc_img)
        mask = np.asarray(mask, dtype="uint8")

        # Convert images to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        if self.enable_mask:
            before_gray = cv2.bitwise_and(before_gray, before_gray, mask=mask)
            after_gray = cv2.bitwise_and(after_gray, after_gray, mask=mask)

        # Compute SSIM between the two images
        (score, diff) = structural_similarity(
                before_gray, after_gray, gaussian_weights=True, full=True)

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1]
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")
        diff_inv = 255 - diff
        return sum(diff_inv.flatten())


if __name__ == "__main__":
    reconstructer = Reconstructer("/mnt/d/reconstruct", "ucsdped")
    reconstructer.run_all()
