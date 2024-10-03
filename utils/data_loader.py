import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.png', '_mask.png'))

        # 画像とマスクの読み込み。両方ともグレースケールで読み込み
        image = Image.open(image_path).convert("L")  # Grayscaleに変換
        mask = Image.open(mask_path).convert("L")  # マスクもGrayscaleに変換

        if self.transform:
            # カスタムTransform関数で画像とマスクに同じ変換を適用
            image, mask = self.transform(image, mask)

        return image, mask
