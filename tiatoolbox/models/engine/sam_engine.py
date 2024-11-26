import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from PIL import Image
from skimage.io import imread as skimread
from torchvision import transforms
from torchvision.models import inception_v3

from tiatoolbox import logger
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchPredictor,
)
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
    WSIStreamDataset,
)

from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils.misc import download_data, imread, select_device
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta, WSIReader

class SamEngine:
    def __init__(self, 
                 model: SAM):
        self.model = model
        # Defining ioconfig
        self.iostate = IOSegmentorConfig(
            input_resolutions=[
                {"units": "mpp", "resolution": 1.0},
            ],
            output_resolutions=[
                {"units": "mpp", "resolution": 1.0},
            ],
            patch_input_shape=[512, 512],
            patch_output_shape=[512, 512],
            stride_shape=[512, 512],
            save_resolution={"units": "mpp", "resolution": 1.0},
        )

    def load_wsi(self, file_name):
        reader = WSIReader.open(file_name)
        self.img = reader.slide_thumbnail(
            resolution=self.iostate.save_resolution["resolution"],
            units=self.iostate.save_resolution["units"],
        )
        return self.img

    def predict(self, file_name, prompts = None, device = "cpu"):
        batch_data = self.load_wsi(file_name)
        self.prediction = self.model.infer_batch(model=self.model, batch_data=batch_data, prompts=prompts, device=device)
        return self.prediction
    
    def display_prediction(self, prediction=None):
        if prediction is None:
            prediction = self.prediction
        plt.figure(figsize=(20, 20))
        plt.imshow(self.img)
        self.show_anns(prediction)
        plt.axis('off')
        plt.show()

    # Imported from SAM2 example Jupyter notebook
    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
        ax.imshow(img)