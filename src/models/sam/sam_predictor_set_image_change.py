import torch
import numpy as np


def new_set_image(
    self,
    image: np.ndarray,
    image_format: str = "RGB",
) -> None:
    """
    Calculates the image embeddings for the provided image, allowing
    masks to be predicted with the 'predict' method.

    Arguments:
        image (np.ndarray): The image for calculating masks. Expects an
        image in HWC uint8 format, with pixel values in [0, 255].
        image_format (str): The color format of the image, in ['RGB', 'BGR'].
    """
    #assert image_format in [
    #    "RGB",
    #    "BGR",
    #], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
    #if image_format != self.model.image_format:
    #    image = image[..., ::-1]

    # Transform the image to the form expected by the model
    #image = torch.as_tensor(image, device=self.device)
    input_image = self.transform.apply_image_torch(image)
    #input_image_torch = torch.as_tensor(input_image, device=self.device)
    #input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_image_torch = (input_image*256).to(torch.uint8)

    #print(input_image_torch.shape, image.shape[-2:])
    self.set_torch_image(input_image_torch, image.shape[-2:])


