import click
from facekit.detectors import MTCNNDetector
from vidsnap.utils import run_video_extractor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from colorama import Fore, Style

import glob
from pathlib import Path
import shutil


@click.group()
def main():
    click.echo("Facekit v1.1.0")


@main.command()
@click.option('--in_path', '-i', type=str, default='./in', help="Path where detector will pick up images.")
@click.option('--out_path', '-o', type=str, default='./out', help="Path where detector will store images.")
@click.option('--accuracy', '-a', type=float, default=0.98, help="Minimum detector threshold accuracy.")
@click.option('--preload', is_flag=True, help="Preload images in memory. "
                                              "More memory intensive, might be faster on HDDs!")
def extract_faces(in_path, out_path, accuracy, preload):
    detector = MTCNNDetector(input_path=in_path, output_path=out_path, accuracy_threshold=accuracy, preload=preload)
    detector.extract_faces()
    detector.store_extracted_faces()


@main.command()
@click.option('--video_in', '-v', type=str, default='./video_in')
@click.option('--video_interval', type=int, default=5)
@click.option('--detector_in', '-i', type=str, default='./detector_in')
@click.option('--detector_out', '-o', type=str, default='./detector_out')
@click.option('--accuracy', '-a', type=float, default=0.98, help="Minimum detector threshold accuracy.")
@click.option('--preload', is_flag=True, help="Preload images in memory. "
                                              "More memory intensive, might be faster on HDDs!")
def extract_faces_video(video_in, video_interval, detector_in, detector_out, accuracy, preload):
    click.secho("Running video extractor, this may take a while...")
    run_video_extractor(input_path=video_in,
                        output_path=detector_in,
                        video_extension='mp4',
                        workers=1,
                        snap_every_x=video_interval)

    detector = MTCNNDetector(input_path=detector_in, output_path=detector_out, accuracy_threshold=accuracy, preload=preload)
    detector.extract_faces()
    detector.store_extracted_faces()


def handle_move_or_copy(image_path, out_dir, copy_mode, classifier_string):
    if copy_mode:
        print(f"{Fore.LIGHTCYAN_EX}Copying{Style.RESET_ALL} {image_path} to {out_dir}/{classifier_string}/{image_path.name}")
        shutil.copy(image_path, Path(f"{out_dir}/{classifier_string}/{image_path.name}"))
    else:
        print(f"{Fore.LIGHTGREEN_EX}Moving{Style.RESET_ALL} {image_path} to {out_dir}/{classifier_string}/{image_path.name}")
        shutil.move(image_path, Path(f"{out_dir}/{classifier_string}/{image_path.name}"))


def classifier_keypress(event, image_path, out_dir, plt, copy):
    print(f'Pressed: {event.key}')
    sys.stdout.flush()
    image_path = Path(image_path)
    if event.key == '1':
        handle_move_or_copy(image_path, out_dir, copy_mode=copy, classifier_string='M')
        plt.close()
    elif event.key == '2':
        handle_move_or_copy(image_path, out_dir, copy_mode=copy, classifier_string='F')
        plt.close()
    elif event.key == '0':
        plt.close()
    elif event.key == 'x':
        print("Shutting down!")
        plt.close()
        exit()


@main.command()
@click.option('--classifier_in', '-i', type=str)
@click.option('--classifier_out', '-o', type=str)
@click.option('--copy', '-c', is_flag=True)
def categorize_gender_manual(classifier_in, classifier_out, copy):
    pin = Path(classifier_in)
    if not pin.exists():
        raise FileNotFoundError("The input path doesn't seem to exist.")

    pout = Path(classifier_out)
    if not pout.exists():
        raise FileNotFoundError("The output root path doesn't seem to exist.")

    path_label_m = pout / 'M'
    path_label_f = pout / 'F'
    if not path_label_m.exists() or not path_label_f.exists():
        raise FileNotFoundError("The label paths don't exist. Make sure your output dir has an M and F folder!")

    for image_path in glob.glob(f"{classifier_in}/*.jpg"):
        fig, ax = plt.subplots()
        ax.set_xlabel('1 = M\n2 = F\nx = Exit')
        img = mpimg.imread(image_path)
        fig.canvas.mpl_connect(
            'key_press_event',
            lambda event: classifier_keypress(event, image_path, classifier_out, plt, copy)
        )
        ax.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()