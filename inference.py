import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet


input_path = "./input/"
output_path = "./output/"
ckpt_path = "./pretrained/modnet_photographic_portrait_matting.ckpt"


def load_model():
    print("Loading modnet...")
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
    modnet.load_state_dict(weights)
    modnet.eval()

    return modnet


def inference_images(modnet, im_name):
    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # read image
    im = Image.open(os.path.join(input_path, im_name))

    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode="area")

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and return matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode="area")
    matte = matte[0][0].data.cpu().numpy()
    matte_img = Image.fromarray(((matte * 255).astype("uint8")), mode="L")

    return matte_img


def extract_img_from_bg(image, matte):
    # calculate display resolution
    w, h = image.width, image.height
    rw, rh = 800, int(h * 800 / (3 * w))

    # obtain predicted foreground
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
    result_img = Image.fromarray(np.uint8(foreground))

    return result_img


def main():
    image_names = os.listdir(input_path)
    model = load_model()
    for image_name in image_names:
        # Load original image
        print(f"Processing {image_name}...")
        try:
            image = Image.open(os.path.join(input_path, image_name))
            # Inference image
            matte_img = inference_images(model, image_name)
            # Combine image
            result_img = extract_img_from_bg(image, matte_img)
            # Save img
            result_img.save(f"{output_path}/{image_name}")
        except:
            print(f"Coundn't process image {image_name}")


if __name__ == "__main__":
    main()
