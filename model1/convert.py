"""Resize and crop images to square, save as tiff."""
from __future__ import division, print_function
import os
from multiprocessing.pool import Pool
import click
import numpy as np
from PIL import Image, ImageFilter
import cv2

N_PROC = 2


def convert(fname, crop_size):
    """
    Convert the image with the location fname to the size crop_size.
    :param fname: string
    :param crop_size: tuple
    :return: array
    """
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original 
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized

scale = 256


def square_bbox(img):
    """
    The first thing that worked well enough was computing a threshold value from the maximum of a thin left
    and right strip of a blurred copy of the image. The bounding box is used to crop the original image.
    """
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def get_convert_fname(fname, extension, directory, convert_directory):
    """
    Replace the filename in the original directory to the filename in the converted directory with the
    given extension.
    :return: New filename
    """
    return fname.replace('jpeg', extension).replace(directory, convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory, 
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        save(img, convert_fname) 


def save(img, fname):
    img.save(fname, quality=97)

# The click module gives options to the user to change the variables during runtime through the command line.

@click.command()
@click.option('--directory', default='/media/azrael/Data/data/train_res_small', show_default=True,
              help="Directory with original images.")
@click.option('--convert_directory', default='/media/azrael/Data/data/train_res_smaller', show_default=True,
              help="Where to save converted images.")
@click.option('--crop_size', default=64, show_default=True,
              help="Size of converted images.")
@click.option('--extension', default='tiff', show_default=True,
              help="Filetype of converted images.")
def main(directory, convert_directory, crop_size, extension):
    """
    Run this function to preprocess the data.
    :param directory: The directory where the original images are present.
    :param convert_directory: The directory where the rescaled images are present.
    :param crop_size: The final size of the images.
    :param extension: The extension of the images.
    :return: None
    """
    try:
        # Form the convert directory if not present.
        os.mkdir(convert_directory)
    except OSError:
        print("The directory can't be formed")
    # Walk through the directory and find the filenames and store them in a list.
    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
                 for f in fn if f.endswith('jpeg') or f.endswith('tiff')] 
    filenames = sorted(filenames)

    print("Resizing images in {} to {}, this takes a while."
          "".format(directory, convert_directory))

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool on my machine
    batchsize = 500
    batches = n // batchsize + 1
    pool = Pool(N_PROC)

    args = []

    for f in filenames:
        args.append((convert, (directory, convert_directory, f, crop_size, 
                               extension)))

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')

if __name__ == '__main__':
    main()


