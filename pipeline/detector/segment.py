import torch, os, cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm


def save_masks_to_disk(masks, mask_dir):
    """Save masks to disk as individual npz files."""
    os.makedirs(mask_dir, exist_ok=True)
    mask_paths = []
    for i, mask in enumerate(masks):
        mask_path = os.path.join(mask_dir, f"mask_{i:08d}.npz")
        np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
        mask_paths.append(mask_path)
    return mask_paths


def load_mask_from_disk(mask_path):
    """Load a single mask from disk."""
    data = np.load(mask_path)
    return data['mask'].astype(np.float32)


def load_masks_from_disk(mask_paths, batch_size=100):
    """Load multiple masks from disk efficiently."""
    masks = []
    for path in mask_paths:
        masks.append(load_mask_from_disk(path))
    return np.array(masks)

def preprocess_fn(img):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    # img_cv2 = cv2.cvtColor(cv2.imread(imgfname), cv2.COLOR_BGR2RGB)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return tf(img)

class ImageFolder(Dataset):
    def __init__(self, images, preprocess_fn=preprocess_fn) -> None:
        super().__init__()
        self.images = images
        self.preprocess_fn = preprocess_fn
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return {
            'img': self.preprocess_fn(self.images[index]),
        }


def segment_preprocess_fn(img, max_length=512):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    w, h = img.size[:2]
    scale = max(np.ceil(max(h, w) / max_length), 1)

    if scale >1:
        img = img.resize((int(w/ scale), int(h / scale)))

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return tf(img)

def segment_subjects(images, device='cuda', max_length=512, mask_dir=None):
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
    )
    segm_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50',
                                weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
    segm_model.eval()

    if isinstance(images[0], str): 
        org_hw = cv2.imread(images[0]).shape[:2]
    else:
        org_hw = images[0].shape[:2]
    
    segm_dataloader = DataLoader(ImageFolder(images, segment_preprocess_fn), batch_size=16, shuffle=False, 
                                    num_workers=8 if os.cpu_count() > 8 else os.cpu_count())
    
    save_to_disk = mask_dir is not None
    if save_to_disk:
        os.makedirs(mask_dir, exist_ok=True)
        print(f"create masks dir {mask_dir}")
        mask_paths = []
    print(f"output mask dir is {mask_dir}")
    
    global_idx = 0
    
    for batch in tqdm(segm_dataloader, desc='Segmenting frames', total=len(segm_dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        with torch.no_grad():
            output = segm_model(batch['img'])['out']
        
        batch_masks = (output.argmax(1) == 15).to(torch.float)
        batch_masks = F.interpolate(batch_masks.unsqueeze(1), size=(
            org_hw[0], org_hw[1]), mode='bilinear', align_corners=True).squeeze(1)
        batch_masks = (batch_masks > 0.1).cpu().numpy()

        if save_to_disk:
            for i in range(len(batch_masks)):
                mask_path = os.path.join(
                    mask_dir, f"mask_{global_idx:08d}.npz")
                np.savez_compressed(
                    mask_path, mask=batch_masks[i].astype(np.uint8))
                mask_paths.append(mask_path)
                global_idx += 1
        else:
            if global_idx == 0:
                all_masks = batch_masks
            else:
                all_masks = np.concatenate([all_masks, batch_masks], axis=0)
            global_idx += len(batch_masks)

        del output, batch_masks
    
    if save_to_disk:
        del segm_model
        torch.cuda.empty_cache()
        return mask_paths
    
    return all_masks