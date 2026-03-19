import cv2 as cv
import numpy as np
import os
import sys


def cartoon_render(
    img,
    median_ksize=5,
    adaptive_blocksize=9,
    adaptive_c=9,
    bilateral_d=9,
    sigma_color=300,
    sigma_space=300,
    color_levels=8,
    edge_thickness=1
):
    """
    Convert an input image into a cartoon-style image.

    Steps
    1) Noise reduction by median filtering
    2) Edge extraction by adaptive thresholding
    3) Edge-preserving smoothing by bilateral filtering
    4) Color quantization to make flat cartoon-like colors
    5) Combine smoothed color image with edge mask
    """

    # 1. Gray-scale conversion
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Median blur for denoising
    # ksize must be odd and >= 3
    median_ksize = max(3, median_ksize)
    if median_ksize % 2 == 0:
        median_ksize += 1
    gray_blur = cv.medianBlur(gray, median_ksize)

    # 3. Adaptive threshold for edge extraction
    # block size must be odd and > 1
    adaptive_blocksize = max(3, adaptive_blocksize)
    if adaptive_blocksize % 2 == 0:
        adaptive_blocksize += 1

    edges = cv.adaptiveThreshold(
        gray_blur,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        adaptive_blocksize,
        adaptive_c
    )

    # 4. Bilateral filter for smooth color while preserving edges
    color = cv.bilateralFilter(
        img,
        bilateral_d,
        sigma_color,
        sigma_space
    )

    # 5. Color quantization (posterization)
    # Make colors flatter, more cartoon-like
    color_levels = max(2, color_levels)
    step = max(1, 256 // color_levels)
    color_quantized = (color // step) * step + step // 2
    color_quantized = np.clip(color_quantized, 0, 255).astype(np.uint8)

    # 6. Make edge lines thicker if needed
    if edge_thickness > 1:
        kernel = np.ones((edge_thickness, edge_thickness), dtype=np.uint8)
        edges = cv.erode(edges, kernel, iterations=1)

    # 7. Combine color image with edge mask
    cartoon = cv.bitwise_and(color_quantized, color_quantized, mask=edges)

    return gray_blur, edges, color_quantized, cartoon


def put_info(img, lines, start=(10, 25), dy=28):
    canvas = img.copy()
    x, y = start
    for line in lines:
        cv.putText(canvas, line, (x, y), cv.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
        cv.putText(canvas, line, (x, y), cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1)
        y += dy
    return canvas


def make_preview(original, gray_blur, edges, color_quantized, cartoon):
    gray_bgr = cv.cvtColor(gray_blur, cv.COLOR_GRAY2BGR)
    edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    h, w = original.shape[:2]
    tiles = []
    for img in [original, gray_bgr, edges_bgr, color_quantized, cartoon]:
        tiles.append(cv.resize(img, (w, h)))

    top = np.hstack((tiles[0], tiles[1], tiles[2]))
    bottom = np.hstack((tiles[3], tiles[4], np.zeros_like(tiles[4])))

    preview = np.vstack((top, bottom))
    return preview


def save_result(output_dir, input_path, cartoon_img, preview_img):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    cartoon_path = os.path.join(output_dir, f"{base}_cartoon.png")
    preview_path = os.path.join(output_dir, f"{base}_preview.png")

    cv.imwrite(cartoon_path, cartoon_img)
    cv.imwrite(preview_path, preview_img)

    return cartoon_path, preview_path


def main():
    if len(sys.argv) < 2:
        print("사용법: python main.py <image_path>")
        print("예시: python main.py input.jpg")
        return

    image_path = sys.argv[1]
    img = cv.imread(image_path)

    if img is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    # Resize for comfortable preview if too large
    max_width = 900
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv.resize(img, None, fx=scale, fy=scale)

    # Initial parameters
    median_ksize = 5
    adaptive_blocksize = 9
    adaptive_c = 9
    bilateral_d = 9
    sigma_color = 250
    sigma_space = 250
    color_levels = 8
    edge_thickness = 1

    window_name = "Cartoon Rendering"
    output_dir = "results"

    while True:
        gray_blur, edges, color_quantized, cartoon = cartoon_render(
            img,
            median_ksize=median_ksize,
            adaptive_blocksize=adaptive_blocksize,
            adaptive_c=adaptive_c,
            bilateral_d=bilateral_d,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
            color_levels=color_levels,
            edge_thickness=edge_thickness
        )

        preview = make_preview(img, gray_blur, edges, color_quantized, cartoon)

        lines = [
            f"Median ksize     : {median_ksize}",
            f"Adaptive block   : {adaptive_blocksize}",
            f"Adaptive C       : {adaptive_c}",
            f"Bilateral d      : {bilateral_d}",
            f"Sigma color      : {sigma_color}",
            f"Sigma space      : {sigma_space}",
            f"Color levels     : {color_levels}",
            f"Edge thickness   : {edge_thickness}",
            "S: save | ESC: quit",
            "[ ]: median | - =: blocksize | , .: C",
            "1 2: sigmaColor | 3 4: sigmaSpace | 5 6: colorLevels | 7 8: edgeThickness",
        ]

        combined = np.hstack((img, cartoon))
        combined = put_info(combined, lines)
        cv.imshow(window_name, combined)

        key = cv.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break

        elif key == ord('s') or key == ord('S'):
            cartoon_path, preview_path = save_result(output_dir, image_path, cartoon, preview_info)
            print("저장 완료:")
            print(" -", cartoon_path)
            print(" -", preview_path)

        elif key == ord('[') or key == ord('{'):
            median_ksize = max(3, median_ksize - 2)

        elif key == ord(']') or key == ord('}'):
            median_ksize += 2

        elif key == ord('-') or key == ord('_'):
            adaptive_blocksize = max(3, adaptive_blocksize - 2)
            if adaptive_blocksize % 2 == 0:
                adaptive_blocksize -= 1

        elif key == ord('=') or key == ord('+'):
            adaptive_blocksize += 2
            if adaptive_blocksize % 2 == 0:
                adaptive_blocksize += 1

        elif key == ord(',') or key == ord('<'):
            adaptive_c = max(0, adaptive_c - 1)

        elif key == ord('.') or key == ord('>'):
            adaptive_c += 1

        elif key == ord('1'):
            sigma_color = max(10, sigma_color - 20)

        elif key == ord('2'):
            sigma_color += 20

        elif key == ord('3'):
            sigma_space = max(10, sigma_space - 20)

        elif key == ord('4'):
            sigma_space += 20

        elif key == ord('5'):
            color_levels = max(2, color_levels - 1)

        elif key == ord('6'):
            color_levels = min(32, color_levels + 1)

        elif key == ord('7'):
            edge_thickness = max(1, edge_thickness - 1)

        elif key == ord('8'):
            edge_thickness = min(7, edge_thickness + 1)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()