import os
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.num_images = len(loader)
        self.preload()

    def preload(self):
        try:
            self.source_img, self.target_img, self.gt_img = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.source_img, self.target_img, self.gt_img = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.source_img  = self.source_img.cuda(non_blocking=True)
            self.source_img  = self.source_img.sub_(self.mean).div_(self.std)
            self.target_img  = self.target_img.cuda(non_blocking=True)
            self.target_img  = self.target_img.sub_(self.mean).div_(self.std)
            self.gt_img      = self.gt_img.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        source_img  = self.source_img
        target_img  = self.target_img
        gt_img      = self.gt_img
        self.preload()
        return source_img, target_img, gt_img

    def __len__(self):
        """Return the number of images."""
        return self.num_images

# -------------------------------------------

class SwappingDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                    base_dir,
                    img_transform,
                    source_subffix='png',
                    target_subffix='png'):
        """Initialize and preprocess the Swapping dataset."""
        self.base_dir       = base_dir
        self.img_transform  = img_transform
        self.source_subffix = source_subffix
        self.target_subffix = target_subffix
        self.source_dataset = []
        self.target_dataset = []
        self.gt_dataset     = []

        self.preprocess()

        self.num_images = len(self.gt_dataset)

    def preprocess(self):
        """Preprocess the Swapping dataset."""
        print("processing Swapping dataset images...")

        name_to_path = {
            'celeba': '/cephFS/gaojie/face_swap/train/celeba',
            'eceleb': '/cephFS/gaojie/face_swap/train/eceleb',
            'model' : '/cephFS/gaojie/face_swap/train/model',
        }

        subffix_jpg_data = ['celeba']
        subffix_png_data = ['eceleb', 'model']

        gt_folders = ['celeba_to_celeba', 'eceleb_to_model', 'model_to_eceleb']
        for folder in gt_folders:
            if not os.path.exists(os.path.join(self.base_dir, folder)):
                print(f"Folder {folder} does not exist in {self.base_dir}. Please check the path.")
                continue

            source_name = folder.split('_')[0]
            target_name = folder.split('_')[2]
            if source_name not in name_to_path or target_name not in name_to_path:
                print(f"Source or target name not found in name_to_path: {source_name}, {target_name}")
                continue

            if source_name in subffix_jpg_data:
                self.source_subffix = 'jpg'
            elif source_name in subffix_png_data:
                self.source_subffix = 'png'

            if target_name in subffix_jpg_data:
                self.target_subffix = 'jpg'
            elif target_name in subffix_png_data:
                self.target_subffix = 'png'

            source_path = name_to_path[source_name]
            target_path = name_to_path[target_name]
            gt_path = os.path.join(self.base_dir, folder)

            gt_names = os.listdir(gt_path)
            for gt_name in gt_names:
                source_name = gt_name.split('.')[0].split('_')[0]
                target_name = gt_name.split('.')[0].split('_')[1]

                self.source_dataset.append(os.path.join(source_path, source_name + '.' + self.source_subffix))
                self.target_dataset.append(os.path.join(target_path, target_name + '.' + self.target_subffix))
                self.gt_dataset.append(os.path.join(gt_path, gt_name))

        print(f'source size: {len(self.source_dataset)}, target size: {len(self.target_dataset)}, gt size: {len(self.gt_dataset)}')

    def __getitem__(self, index):
        """Return pairs (source, target, gt)."""

        source_name = self.img_transform(Image.open(self.source_dataset[index]))
        target_name = self.img_transform(Image.open(self.target_dataset[index]))
        gt_name     = self.img_transform(Image.open(self.gt_dataset[index]))

        return source_name, target_name, gt_name

    def __len__(self):
        """Return the number of images."""
        return self.num_images

# -------------------------------------------

def GetLoader(dataset_roots,
              batch_size=16,
              dataloader_workers=8
              ):
    """Build and return a data loader."""

    num_workers         = dataloader_workers
    data_root           = dataset_roots

    c_transforms = []
    # c_transforms.append(T.Resize((256, 256)))
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    content_dataset = SwappingDataset(data_root, c_transforms)
    content_data_loader = data.DataLoader(dataset=content_dataset, batch_size=batch_size,
                                          drop_last=True, shuffle=True,
                                          num_workers=num_workers, pin_memory=True)

    prefetcher = data_prefetcher(content_data_loader)
    return prefetcher
