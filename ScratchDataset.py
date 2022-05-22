import torch
from torch.utils.data import Dataset
import glob
from PIL import Image, ImageDraw
import json
from torchvision import transforms
import torchvision.transforms.functional as F

from torchvision.io import read_image, ImageReadMode

class ScratchDataset(Dataset):
    def __init__(self, dir = '/home/macenrola/Hug_2Tb/DamageDetection/carScratchDetector/train', size=(224,224)):
        """
        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset
        """
        super().__init__()
        self.img_size = size
        self.images = glob.glob(f'{dir}/image*')
        self.annotation_file = glob.glob(f'{dir}/*json')[0]
        print(self.annotation_file)
        self.annotations = self.load_annotations()
        self.transform   = transforms.Compose([
                            transforms.ToPILImage(),
                            SquarePad(),
                            transforms.Resize(self.img_size),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=.5, hue=.3),
                            transforms.RandomEqualize(),
                            transforms.ToTensor()])

        self.reading_mode = ImageReadMode(ImageReadMode.RGB)

    def load_annotations(self):
        annotations = {}
        with open(self.annotation_file, "r") as read_file:
            data = json.load(read_file)
            for d in data.values():
                annotations[d['filename']]=d['regions']
        return annotations

    def return_mask(self, img, regions):
        mask = torch.zeros(img.shape)
        mask = F.to_pil_image(mask)
        img1 = ImageDraw.Draw(mask)
        for r in regions:
            img1.polygon(list(zip(regions[r]['shape_attributes']['all_points_x'], regions[r]['shape_attributes']['all_points_y'])), fill ="white", outline ="white")
        img = F.to_tensor(mask)
        return img

    def return_bbox(self, regions):
        bboxes = torch.zeros([len(regions),4])
        bboxes_plain = []
        for i, r in enumerate(regions):
            bboxes[i,:] =   torch.tensor([
                            min(regions[r]['shape_attributes']['all_points_x']),
                            min(regions[r]['shape_attributes']['all_points_y']),
                            max(regions[r]['shape_attributes']['all_points_x']),
                            max(regions[r]['shape_attributes']['all_points_y'])
                            ])
            bboxes_plain.append((
            min(regions[r]['shape_attributes']['all_points_x']),
            min(regions[r]['shape_attributes']['all_points_y']),
            max(regions[r]['shape_attributes']['all_points_x']),
            max(regions[r]['shape_attributes']['all_points_y'])
            ))
        return bboxes_plain

    def draw_bboxes(self, img, bboxes):
        img = F.to_pil_image(img)
        img_modified=img.copy()

        img1 = ImageDraw.Draw(img_modified)
        for bbox in bboxes:
            img1.rectangle(bbox, width=1)
        img = F.to_tensor(img_modified)
        return img

    def resize_bbox(self, bbox, previous_img_sz, current_img_size, square_padded=True):
        x0, y0, x1, y1 = bbox
        if square_padded:
            max_wh = max(previous_img_sz)
            p_left, p_top = [(max_wh - s) // 2 for s in previous_img_sz]
            p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(previous_img_sz, [p_left, p_top])]
            padding = (p_left, p_top, p_right, p_bottom) ### y and x get interverted here
            x0, y0, x1, y1 = x0+p_top, y0+p_left, x1+p_top, y1+p_left
            previous_img_sz = (max_wh,max_wh)

        x0_rel, y0_rel, x1_rel, y1_rel = x0/previous_img_sz[1], y0/previous_img_sz[0], x1/previous_img_sz[1], y1/previous_img_sz[0]

        x0_resized = int(x0_rel*current_img_size[1])
        y0_resized = int(y0_rel*current_img_size[0])
        x1_resized = int(x1_rel*current_img_size[1])
        y1_resized = int(y1_rel*current_img_size[0])



        return (x0_resized, y0_resized, x1_resized, y1_resized)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = read_image(self.images[idx], mode=self.reading_mode)
        original_img_size = img.shape[1:]
        mask = self.return_mask(img, self.annotations[self.images[idx].split('/')[-1]])
        bboxes = self.return_bbox( self.annotations[self.images[idx].split('/')[-1]])
        img_bbox = self.draw_bboxes(img, bboxes)


        Xi = self.transform(img)
        img_resized_bboxes = [self.resize_bbox(bbox, original_img_size, self.img_size, square_padded=True) for bbox in bboxes]
        boxes = torch.as_tensor(img_resized_bboxes, dtype=torch.float32)
        labels = [0]*len(img_resized_bboxes)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # yi = self.transform(mask)

        return Xi, {'boxes':boxes, 'labels':labels}

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


def main():
    import matplotlib.pyplot as plt
    scratch_ds = ScratchDataset( dir = '/home/macenrola/Hug_2Tb/DamageDetection/carScratchDetector/train')
    for scratch in scratch_ds:
        Xi, yi = scratch
        print(Xi, yi)
        plt.imshow(Xi.permute(1, 2, 0))
        plt.show()

        img_resized_bboxed = scratch_ds.draw_bboxes(Xi, yi['boxes'].tolist())
        plt.imshow(img_resized_bboxed.permute(1, 2, 0))
        plt.show()

if __name__ == "__main__":
    main()
