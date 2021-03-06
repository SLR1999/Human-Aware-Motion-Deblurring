import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
import glob

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 


class DeblurrDataset(Dataset):
    def __init__(self, blurred_image_path, real_image_path, attention_path, transform=None):
        self.blurred_image_path = blurred_image_path
        self.real_image_path = real_image_path
        self.attention_path = attention_path
        self.transform = transform
        self._add_images()

    def _add_images(self):
        print(self.blurred_image_path)
        print(self.attention_path)
        print(self.real_image_path)
        self.blurred_images = glob.glob(self.blurred_image_path)
        print(len(self.blurred_images))
        set_of_blurred_images = len(self.blurred_images)
        self.attention_maps = glob.glob(self.attention_path)
        print(len(self.attention_maps))
        self.real_images = glob.glob(self.real_image_path)
        total_ground_truth = len(self.real_images)
        print(len(self.real_images))
        self.real_images = self.real_images*int((set_of_blurred_images/total_ground_truth))

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        blurred_img_name = self.blurred_images[idx]
        real_image_name = self.real_images[idx]
        attention_map_name = self.attention_maps[idx]
        blurred_image = io.imread(blurred_img_name)
        real_image = io.imread(real_image_name)
        attention_map = io.imread(attention_map_name)

        if self.transform: 
            blurred_image = self.transform(blurred_image)
            real_image = self.transform(real_image)
            # attention_map = self.transform(attention_map)
        attention_map = torch.Tensor(attention_map)/255
        attention_width, attention_height = list(attention_map.size()   )
        attention_map = attention_map.view(1,attention_width, attention_height)
        return (blurred_image, real_image, attention_map)


if __name__ == "__main__":
    d = DeblurrDataset("/home/ananya/Documents/de blurring/Human-Aware-Motion-Deblurring/data/train/blurred_images/",
                               "/home/ananya/Documents/de blurring/Human-Aware-Motion-Deblurring/data/train/clear_images/",
                               "/home/ananya/Documents/de blurring/Human-Aware-Motion-Deblurring/data/train/attention_maps/",
                               transform)
    print(len(d))