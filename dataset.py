from torch import Generator
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms


class MyDataset(Dataset):
    def __init__(self, root = './data', train=True, patch_size=100, bag_size=36):
        self.train = train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            self.Patchify(patch_size),
        ])
        self.dataset = random_split(
            dataset=datasets.ImageFolder(root=root, transform=self.transform),
            lengths=[0.2, 0.8],
            generator=Generator().manual_seed(0)
        )[self.train]

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    class Patchify(object):
        def __init__(self, patch_size=100):
            self.patch_size = patch_size

        def __call__(self, img):
            c, h, w = img.shape
            img = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
            img = img.permute(1, 2, 0, 3, 4)
            img = img.contiguous().view(-1, c, self.patch_size, self.patch_size)
            return img


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    dataset = MyDataset(train=False)
    dataloader = DataLoader(dataset)

    for i, (X, y) in enumerate(dataloader):
        img = make_grid(X[0], nrow=6)
        img = img.permute(1, 2, 0)
        img = img * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(img)
        fig.savefig(f'img/{y[0]}_{i}')
        plt.close(fig)