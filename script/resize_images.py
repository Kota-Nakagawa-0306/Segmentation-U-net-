import os
import cv2

def clear_directory(directory):
    """ディレクトリ内のすべてのファイルを削除"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def split_and_resize_image(source_path, target_dir, num_splits=9, scale_factor=10, interpolation=cv2.INTER_LANCZOS4):
    """画像を分割し、解像度を倍にリサイズして保存する"""

    # 保存先のディレクトリ内をクリア
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        clear_directory(target_dir)  # 既存ファイルを削除

    # 元画像を読み込む
    img = cv2.imread(source_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image: {source_path}")
        return

    # 画像の高さと幅を取得
    height, width = img.shape[:2]
    split_side = int(num_splits ** 0.5)

    # 画像を分割してリサイズ
    count = 1
    for i in range(split_side):
        for j in range(split_side):
            part = img[i * (height // split_side): (i + 1) * (height // split_side),
                       j * (width // split_side): (j + 1) * (width // split_side)]

            # 解像度を2倍にリサイズ
            new_size = (int(part.shape[1] * scale_factor), int(part.shape[0] * scale_factor))
            resized_part = cv2.resize(part, new_size, interpolation=interpolation)

            # PNG形式で保存 (非圧縮)
            target_path = os.path.join(target_dir, f"image{count}_resized.png")
            cv2.imwrite(target_path, resized_part, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 圧縮なしで保存
            print(f"Saved resized image: {target_path}")
            count += 1


# 使用例
source_image_path = '../data/source_images/clear_image.jpg'  # 元画像のパス
target_images_dir = '../data/images/'  # リサイズ後の画像を保存するディレクトリ

# 保存前にディレクトリ内の既存ファイルを削除し、画像を分割して解像度を倍にリサイズして保存
split_and_resize_image(source_image_path, target_images_dir, num_splits=9, scale_factor=10)
