import argparse
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocksize", "-b", help="Blocksize", type=int)
    parser.add_argument("--constant", "-c", help="Constant", type=int)
    args = parser.parse_args()

    image = cv2.imread("00-initial.png")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = img_gray
    image = cv2.dilate(image, np.ones((7, 7), np.uint8))
    img_bg = cv2.medianBlur(image, 21)
    image = 255 - cv2.absdiff(img_gray, img_bg)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, args.blocksize, args.constant)

    cv2.imwrite("10-augmented.png", image)


if __name__ == "__main__":
    main()
