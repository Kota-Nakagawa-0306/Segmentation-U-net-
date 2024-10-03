import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet  # モデルのインポート
from utils.data_loader import SegmentationDataset  # データセットのインポート
from torchvision import transforms
import matplotlib.pyplot as plt

# カスタムTransformクラスの定義
class Transform:
    def __init__(self):
        # 画像とマスクに適用する共通の変換を定義
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __call__(self, image, mask):
        # 画像とマスクに同じリサイズ処理を適用
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask


# 予測結果の可視化関数
def visualize_predictions(model, dataloader, device):
    model.eval()  # 評価モードに切り替え
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # 予測
            outputs = model(images)
            preds = torch.sigmoid(outputs)  # 出力にシグモイドを適用
            preds = (preds > 0.3).float()  # 閾値を調整してみる（例: 0.3）

            # CPUに戻してnumpy配列に変換
            images = images.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            masks = masks.cpu().numpy()
            preds = preds.cpu().numpy()

            # 予測された値を出力して確認
            print(f"Predicted min value: {preds.min()}, max value: {preds.max()}")

            # 1つのサンプルを表示
            for i in range(len(images)):
                plt.figure(figsize=(15, 5))

                # 入力画像の表示
                plt.subplot(1, 3, 1)
                plt.imshow(images[i].squeeze(), cmap='gray')
                plt.title("Input Image")

                # 真のマスクの表示
                plt.subplot(1, 3, 2)
                plt.imshow(masks[i].squeeze(), cmap='gray')
                plt.title("True Mask")

                # 予測されたマスクの表示
                plt.subplot(1, 3, 3)
                plt.imshow(preds[i].squeeze(), cmap='gray')
                plt.title("Predicted Mask")

                plt.show()

            break  # 最初のバッチだけを表示


# 学習損失の可視化
def plot_loss_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


# デバイスの設定 (CUDAが利用可能か確認し、なければCPUを使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセットとデータローダーの準備
if __name__ == '__main__':
    # カスタムTransformを使用して画像とマスクに同じ変換を適用
    transform = Transform()
    dataset = SegmentationDataset('data/images', 'data/masks', transform=transform)

    def collate_fn(batch):
        # Noneが含まれている場合はスキップする
        batch = [b for b in batch if b[0] is not None and b[1] is not None]
        if len(batch) == 0:
            return None, None
        return torch.utils.data.default_collate(batch)

    # バッチサイズを1に設定し、num_workersを0にする
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # モデル、損失関数、最適化関数の定義
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習損失を保存するリスト
    epoch_losses = []

    # モデルのトレーニング
    for epoch in range(500):
        epoch_loss = 0.0
        for images, masks in dataloader:
            if images is None or masks is None:
                continue  # スキップ

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # 通常の順伝播と逆伝播
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()

            # 勾配がNoneではないことを確認する
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Gradients for {name} are None")
                else:
                    print(f"Gradients for {name}: {param.grad.norm()}")  # 勾配のノルムを表示

            optimizer.step()

            epoch_loss += loss.item()

        # エポックごとの損失をリストに追加
        avg_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    # 損失の推移をグラフ化
    plot_loss_curve(epoch_losses)

    # モデルの保存
    torch.save(model.state_dict(), 'results/checkpoints/unet_model.pth')
    print("Model saved successfully.")

    # モデルを読み込んで予測を表示
    model.load_state_dict(torch.load('results/checkpoints/unet_model.pth'))
    visualize_predictions(model, dataloader, device)
