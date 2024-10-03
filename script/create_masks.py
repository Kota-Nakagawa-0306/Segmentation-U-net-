import os
import json
import cv2
import numpy as np


def create_mask_from_json(json_path, image_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    return mask


def main():
    input_dir = 'C:/Users/saxph/PycharmProjects/segmentation/data/images'  # 入力画像のディレクトリ
    json_dir = 'C:/Users/saxph/PycharmProjects/segmentation/data/annotations'  # アノテーション（JSONファイル）のディレクトリ
    output_dir = 'C:/Users/saxph/PycharmProjects/segmentation/data/masks'  # 出力マスク画像のディレクトリ

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # JSONファイルが存在する画像に対してのみ処理を行う
    for json_filename in os.listdir(json_dir):
        if json_filename.endswith('.json'):
            # 対応する画像ファイルのパスを取得
            image_filename = json_filename.replace('.json', '.png')
            image_path = os.path.join(input_dir, image_filename)
            json_path = os.path.join(json_dir, json_filename)

            if os.path.exists(image_path):
                print(f'Processing {image_filename}...')
                image = cv2.imread(image_path)
                mask = create_mask_from_json(json_path, image.shape)

                # マスク画像の保存
                mask_output_path = os.path.join(output_dir, image_filename.replace('.png', '_mask.png'))
                cv2.imwrite(mask_output_path, mask)
                print(f'Saved mask: {mask_output_path}')
            else:
                print(f'Image file not found for {json_filename}')


if __name__ == '__main__':
    main()
