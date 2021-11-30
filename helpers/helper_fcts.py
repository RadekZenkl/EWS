from torch.nn.functional import one_hot
from torchvision.transforms.functional import rotate, center_crop, hflip, vflip, normalize, to_tensor, crop
from torch.nn.functional import interpolate, pad
from torchvision.transforms import ColorJitter
from torchvision.transforms import InterpolationMode
import torch

import numpy as np
from torch.utils.data import Dataset
from methods import base


class EWS_Dataset(Dataset):
    def __init__(self, image_paths, transforms):
        super().__init__()
        self.transforms = transforms
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        label_path = (self.image_paths[idx]).replace('.png', '_mask.png')

        image, metadata = base.BenchmarkMethod.read_image(image_path, rgb=True, metadata=True)
        mask = base.BenchmarkMethod.read_image(label_path, rgb=False)

        sample = {}

        iso = np.float(metadata['iso'])
        iso = np.log2(iso / 100)

        fnumber = metadata['fnumber']
        if '/' in fnumber:
            num, den = fnumber.split('/')
            fnumber = np.float((float(num) / float(den)))
        else:
            fnumber = np.float(fnumber)

        exposure = metadata['exposure']
        if '/' in exposure:
            num, den = exposure.split('/')
            exposure = np.float((float(num) / float(den)))
        else:
            exposure = np.float(exposure)

        sample.update({'iso': iso, 'fnumber': fnumber, 'exposure': exposure})
        image_pair = self.transforms({'image': image, 'mask': mask})
        sample.update(image_pair)

        x = {key: torch.as_tensor(sample[key]).float() for key in ['image', 'iso', 'fnumber', 'exposure']}
        y = {key: torch.as_tensor(sample[key]).float() for key in ['mask']}

        return x, y


class PadInput(object):
    """ Pad Tensors to desired shape.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        image = sample['image']
        input_size = (image.shape[-2], image.shape[-1])

        if input_size[0] >= self.output_size[0] or input_size[1] >= self.output_size[1]:
            raise Exception("Larger Input size than Output size before padding")

        if (self.output_size[0] - input_size[0]) % 2 != 0 or (self.output_size[1] - input_size[1]) % 2 != 0:
            raise Exception("Unsupported padding size for reflecting")

        n_pads1 = int((self.output_size[0] - input_size[0]))
        n_pads2 = int((self.output_size[1] - input_size[1]))

        pads = (0, n_pads1, 0, n_pads2)

        sample['image'] = pad(image, pads, 'constant', 0)

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[1], image.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sample['image'] = crop(sample['image'], top, left, new_h, new_w)

        if 'mask' in sample.keys():
            sample['mask'] = crop(sample['mask'], top, left, new_h, new_w)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample['image'] = to_tensor(sample['image'])

        if 'mask' in sample.keys():
            sample['mask'] = torch.round(to_tensor(sample['mask']))

        return sample


class EncodedToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample['image'] = to_tensor(sample['image'])

        if 'mask' in sample.keys():
            sample['mask'] = torch.round(torch.from_numpy(np.array(sample['mask'])).float()).unsqueeze(0)

        return sample


class RandomRotate(object):
    """Rotate randomly the image in a sample.

    Args:
        angle (int): Desired maximal rotation angle. Integer angle in [0, angle] will be sampled at random
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        angle = np.random.randint(0, self.angle)
        sample['image'] = rotate(sample['image'], angle, resample=None, expand=False, center=None, fill=None,
                                 interpolation=InterpolationMode.BILINEAR)

        if 'mask' in sample.keys():
            sample['mask'] = rotate(sample['mask'], angle, resample=None, expand=False, center=None, fill=None,
                                    interpolation=InterpolationMode.NEAREST)

        return sample


class CenterCrop(object):
    """Crop image and mask around the center.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        sample['image'] = center_crop(sample['image'], self.output_size)

        if 'mask' in sample.keys():
            sample['mask'] = center_crop(sample['mask'], self.output_size)

        return sample


class RandomHorizontalFlip(object):
    """ Randomly flip horizontally the image and mask in a sample.

    """

    def __init__(self):
        pass

    def __call__(self, sample):
        if torch.rand(1) >= 0.5:
            sample['image'] = hflip(sample['image'])

            if 'mask' in sample.keys():
                sample['mask'] = hflip(sample['mask'])
        else:
            pass

        return sample


class RandomVerticalFlip(object):
    """ Randomly flip vertically the image and mask in a sample.

    """

    def __init__(self):
        pass

    def __call__(self, sample):
        if torch.rand(1) >= 0.5:
            sample['image'] = vflip(sample['image'])

            if 'mask' in sample.keys():
                sample['mask'] = vflip(sample['mask'])
            if 'weights' in sample.keys():
                sample['weights'] = vflip(sample['weights'])
        else:
            pass

        return sample


class ColorsJitter(object):
    """Apply random color jitter to the image

    Args:
        brightness, contrast, saturation, hue
        for details see original torchvision.transforms.ColorJitter
    """

    def __init__(self, brightness, contrast, saturation, hue):
        self.ColorJitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample['image'] = self.ColorJitter(sample['image'])

        return sample


class Normalize(object):
    """Normalize image.

    Args:
        self, tensor, mean, std, inplace=False
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        sample['image'] = normalize(sample['image'], self.mean, self.std, self.inplace)

        return sample


class Scale(object):
    """
    Upsample image and mask tensors to higher resolution, image is upscale with bilinear interpolation and maks with
    nearest neighbour
    :param scale_factor
    """

    def __init__(self, scale_factor):
        self.factor = scale_factor

    def __call__(self, sample):
        sample['image'] = interpolate(sample['image'].unsqueeze(0), scale_factor=self.factor,
                                      mode='bilinear', align_corners=False)
        sample['image'] = sample['image'].squeeze(0)

        if 'mask' in sample.keys():
            sample['mask'] = interpolate(sample['mask'].unsqueeze(0), scale_factor=self.factor, mode='nearest')
            sample['mask'] = sample['mask'].squeeze(0)
        if 'weights' in sample.keys():
            sample['weights'] = interpolate(sample['weights'].unsqueeze(0), scale_factor=self.factor,
                                            mode='bilinear', align_corners=False)
            sample['weights'] = sample['weights'].squeeze(0)

        return sample


class ScaleImage(object):
    """
    Upsample Image only and leave mask as it is
    :param scale_factor
    """

    def __init__(self, scale_factor):
        self.factor = scale_factor

    def __call__(self, sample):
        sample['image'] = sample['image'].unsqueeze(0)

        sample['image'] = interpolate(sample['image'], scale_factor=self.factor, mode='bilinear', align_corners=False)
        sample['image'] = sample['image'].squeeze(0)

        return sample


class GaussianNoise(object):
    """
    Add gaussian Noise to tensor dictionary
    """

    def __init__(self, sigma, mean=0):
        """
        :param sigma: either a single float or a float tuple for every image channel.
        """
        self.variance = sigma ** 0.5
        self.mean = mean

    def __call__(self, sample):
        sample['image'] = sample['image'] + torch.rand(1) * self.variance * torch.randn_like(sample['image']) + \
                          self.mean

        return sample


def iou(key, outputs, Y, device, pos_id=None):
    """
    Compute intersection over union
    Args:
        key (str): name of the desired output key
        outputs (dict): dictionary with outputs
        Y (dict): dictionary with targets
        device (str): device indicates whether it is run on gpu or cpu
        pos_id (int): wrt. which id the metric should be calculated

    Returns:
        result (batchsize x float): intersection over union, NaN or Inf when class is not present
    """

    _, preds = torch.max(outputs[key], 1)
    preds.to(device)
    Y[key].to(device)

    # Compute mean IoU
    if pos_id is None:
        # one hot encode labels, new dimension gets appended at the end
        preds_one = one_hot(preds)
        Y_b_one = one_hot(Y[key].squeeze(1).long())

        preds_one = preds_one.permute(0, 3, 1, 2)
        Y_b_one = Y_b_one.permute(0, 3, 1, 2)

        intersection = (preds_one & Y_b_one).float().sum((2, 3))
        union = (preds_one | Y_b_one).float().sum((2, 3))

        result = (intersection / union).mean(dim=1)

    # Compute IoU wrt. to given label
    else:
        preds_b = preds.int() == pos_id
        Y_b = Y[key].squeeze(1).int() == pos_id
        intersection = (preds_b & Y_b).float().sum((1, 2))
        union = (preds_b | Y_b).float().sum((1, 2))

        result = intersection / union

    return result  # average across the batch, resp. along different classes
