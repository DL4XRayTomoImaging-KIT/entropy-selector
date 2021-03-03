from tqdm.auto import tqdm
import os
from glob import glob
import re

from matplotlib import pyplot as plt

from concert.readers import TiffSequenceReader, SequenceReaderError


from skimage.morphology import opening
from skimage.measure import label
from skimage.feature import canny
from skimage.transform import rescale
import numpy as np
from skimage.exposure import equalize_adapthist, equalize_hist

norm_to_one = lambda x: (x - x.min()) / (x.max() - x.min())

def robust_norm_correct(img):
    mi = np.percentile(img, 0.5)
    ma = np.percentile(img, 99.5)
    img = norm_to_one(np.clip(img, mi, ma))
    return img

def get_embryo_thresholds(img, canny_alpha=0.75,
                          rescale_coefficient=0.1,
                          threshold=0.3,
                          do_selection=True,
                          line_removal=False):
    img = robust_norm_correct(img)
    img = canny(img, canny_alpha).astype(np.float)
    img = rescale(img, rescale_coefficient, order=0, multichannel=False)
    img = img > threshold
    img = opening(img)

    if line_removal:
        lines_to_remove = np.where(img.mean(1) == 1)
        img[lines_to_remove] = 0

    if do_selection:
        img = label(img)
        area, count = np.unique(img, return_counts=True)
        if len(count) > 1:
            biggest_area = area[area!=0][np.argmax(count[area!=0])]
        else:
            biggest_area = 0
        img = img == biggest_area

    return img

def top_down_embryo_lines(img):
    is_sure_embryo = np.where(get_embryo_thresholds(img).sum(1) > 10)[0]
    return is_sure_embryo[0], is_sure_embryo[-1]

def plot_one_debugging_sample(image, canny_alpha=0.2, rescale_coefficient=0.1,
                              threshold=0.3, do_selection=True, line_removal=False, plot_to_file=None):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    ax1.imshow(equalize_adapthist(image, clip_limit=0.1), cmap='gray')
    ax2.imshow(get_embryo_thresholds(image,
                                     canny_alpha=canny_alpha,
                                     rescale_coefficient=rescale_coefficient,
                                     threshold=threshold,
                                     do_selection=do_selection,
                                     line_removal=line_removal))
    if plot_to_file is not None:
        plt.savefig(plot_to_file)
    else:
        plt.show()

recto_by_coords = lambda c: plt.Rectangle((c[1][0], c[0][0]),
                                          width=c[1][1]-c[1][0],
                                          height=c[0][1]-c[0][0],
                                          color='r', fill=False)

def get_crops_by_mask(mask, expansion=0):
    crop_thr_y = (max(np.where(mask.any(0))[0][0] - expansion, 0),
                  min(np.where(mask.any(0))[0][-1] + expansion, mask.shape[1]))
    crop_thr_x = (max(np.where(mask.any(1))[0][0] - expansion, 0),
                  min(np.where(mask.any(1))[0][-1] + expansion, mask.shape[0]))
    return crop_thr_x, crop_thr_y

def get_two_crops(sequence_reader,
                  canny_alpha=0.2,
                  rescale_coefficient=0.1,
                  threshold=0.3,
                  do_selection=True,
                  line_removal=False):
    total_frames = sequence_reader.num_images
    frame_00 = sequence_reader.read(0)
    frame_90 = sequence_reader.read(int(total_frames/2))

    mask_00 = rescale(get_embryo_thresholds(frame_00,
                                            canny_alpha=canny_alpha,
                                            rescale_coefficient=rescale_coefficient,
                                            threshold=threshold,
                                            do_selection=do_selection,
                                            line_removal=line_removal), 1/rescale_coefficient, order=0)
    mask_90 = rescale(get_embryo_thresholds(frame_90,
                                            canny_alpha=canny_alpha,
                                            rescale_coefficient=rescale_coefficient,
                                            threshold=threshold,
                                            do_selection=do_selection,
                                            line_removal=line_removal), 1/rescale_coefficient, order=0)

    x_00, y_00 = get_crops_by_mask(mask_00)
    x_90, y_90 = get_crops_by_mask(mask_90)

    final_x = (min(x_00[0], x_90[0]), max(x_00[1], x_90[1]))

    return (final_x, y_00, y_90)

def plot_15_with_bb(images_list, names_list, multiboxes_list, meta_information='', plot_to_file=None):
    fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(25, 15))

    for i, (image, name, box) in tqdm(enumerate(zip(images_list, names_list, multiboxes_list))):
        ca = axes[i//5][i%5]
        ca.set_title(name)
        ca.imshow(equalize_hist(image), cmap='gray')
        ca.add_patch(recto_by_coords((box[0], box[1])))
        ca.set_xticks([])
        ca.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle(meta_information)
    if plot_to_file is not None:
        plt.savefig(plot_to_file)
    else:
        plt.show()

get_volume_name = lambda addr: addr.split('/')[-2]
get_meta_str = lambda **kwargs: str(kwargs)

def plot_random_15(images_list, canny_alpha=0.2, rescale_coefficient=0.1,
                              threshold=0.3, do_selection=True, line_removal=False, plot_to_file=None):
    list_of_frames_to_use = np.random.choice(np.array(images_list), size=15)

    images_list, names_list, multiboxes_list = [], [], []
    for i, volume_addr in tqdm(enumerate(list_of_frames_to_use), total=len(list_of_frames_to_use)):
        try:
            seq_reader = TiffSequenceReader(list_of_frames_to_use[i])
            names_list.append(get_volume_name(list_of_frames_to_use[i]))
            images_list.append(seq_reader.read(0))
            multiboxes_list.append(get_two_crops(seq_reader, canny_alpha,
                                                rescale_coefficient=rescale_coefficient,
                                                threshold=threshold,
                                                do_selection=do_selection,
                                                line_removal=line_removal))
        except SequenceReaderError as e:
            print(f'empty sequence at {list_of_frames_to_use[i]}, moving on')

    meta_information = get_meta_str(canny_alpha=canny_alpha, rescale_coefficient=rescale_coefficient, threshold=threshold, do_selection=do_selection, line_removal=line_removal)

    plot_15_with_bb(images_list, names_list, multiboxes_list, meta_information=meta_information, plot_to_file=plot_to_file)

def produce_files_with_thresholds(address_list, canny_alpha=0.2, rescale_coefficient=0.1,
                              threshold=0.3, do_selection=True, line_removal=False, rewrite=False, filename='embryo_area.txt', observe_filename=None):

    images_list, names_list, multiboxes_list = [], [], []

    for volume_addr in tqdm(address_list):
        path_to_save = os.path.abspath(os.path.join(volume_addr, '..', filename))

        if os.path.exists(path_to_save):
            #previously_done.append(volume_addr)
            if not rewrite:
                continue

        try:
            projections_loader = TiffSequenceReader(volume_addr)
        except SequenceReaderError as err:
            print(f'Empty sequence at {volume_addr}, moving on')
            continue

        names_list.append(get_volume_name(volume_addr))
        multi_bbox = get_two_crops(projections_loader, canny_alpha,
                                   rescale_coefficient=rescale_coefficient,
                                   threshold=threshold,
                                   do_selection=do_selection,
                                   line_removal=line_removal)
        images_list.append(projections_loader.read(0))
        multiboxes_list.append(multi_bbox)

        with open(path_to_save, 'w') as f:
            print(str(multi_bbox[0])[1:-1], file=f)
            print(str(multi_bbox[1])[1:-1], file=f)
            print(str(multi_bbox[2])[1:-1], file=f)

    if observe_filename is not None:
        observe_filename=observe_filename.split('.')
        observe_filename[0] += '_{}'
        observe_filename = '.'.join(observe_filename)

        meta_information = get_meta_str(canny_alpha=canny_alpha, rescale_coefficient=rescale_coefficient, threshold=threshold, do_selection=do_selection, line_removal=line_removal)

        for partition in range(int(np.ceil(len(images_list)/15))):
            plot_15_with_bb(images_list[partition*15:(partition+1)*15],
                            names_list[partition*15:(partition+1)*15],
                            multiboxes_list[partition*15:(partition+1)*15],
                            meta_information=meta_information,
                            plot_to_file=observe_filename.format(partition))

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Detector of samples with high entropy')
    # actions
    parser.add_argument('-produce-one-debugging-plot', const=True, default=False,  action='store_const', help='produce one plot with non-tresholded entropy detection')
    parser.add_argument('-produce-random-15-plots', const=True, default=False,  action='store_const',  help='produce plot with random 15 samples')
    parser.add_argument('-generate-treshold-files', const=True, default=False,  action='store_const',  help='generate file called embryo_area_cottected.txt in each working directory')

    # execution parameters
    parser.add_argument('--input-paths', help='either wildcarded paths list or direct path to one very folder or path to the file with paths stored')
    parser.add_argument('--input-regex', default=None, help='regexp which should be matched only by files to proceed. Used as additional filter for the paths')
    parser.add_argument('--plot-to-file', default=None,  help='address of file to save plots if not plotted directly')
    parser.add_argument('--output-file', default='embryo_area.txt',  help='name of file to put selected tresholds')
    parser.add_argument('--rewrite', const=True, default=False, action='store_const', help='If used, will rewrite the previously generated embryo_area_corrected.txt files.')

    # algorithm parameters
    parser.add_argument('--gaussian-sigma', default=0.2, type=float, help='Sigma for the Gaussian filter before the Canny operator. The more is sigma, the less recall detector have. Default value 0.2.')
    parser.add_argument('--rescale-coefficient', type=float, default=0.1, help='Coefficient of the image rescaling for speed. Used as additional noise removal. Default value 0.1.')
    parser.add_argument('--threshold', default=0.3, type=float, help='Just thresholding value for the pixel to be considered as part of the embryo. Default value 0.3')
    parser.add_argument('--line-removal', const=True, default=False, action='store_const', help='If used will remove the solid lines selected as embryo from processing. Can help with specific noises.')

    args = parser.parse_args()

    if args.input_paths.endswith('.txt'):
        filenames = open(args.input_paths).read().splitlines()
    else:
        filenames = glob(args.input_paths)
        if args.input_regex is not None:
            reg_filter = re.compile(args.input_regex)
            filenames = [f for f in filenames if reg_filter.match(f) is not None]

    algo_params = {'canny_alpha': args.gaussian_sigma,
                   'rescale_coefficient': args.rescale_coefficient,
                   'threshold': args.threshold,
                   'line_removal': args.line_removal}

    if args.produce_one_debugging_plot:
        print('plotting the debugging plot of the first projection of first volume in input list')

        image = TiffSequenceReader(filenames[0]).read(0)
        plot_one_debugging_sample(image, **algo_params, plot_to_file=args.plot_to_file)

    if args.produce_random_15_plots:
        print('generating bounding box plots of random 15 samples')

        plot_random_15(filenames, **algo_params, plot_to_file=args.plot_to_file)

    if args.generate_treshold_files:
        print('generating output files for each folder listed')
        produce_files_with_thresholds(filenames, **algo_params, rewrite=args.rewrite, filename=args.output_file, observe_filename=args.plot_to_file)
