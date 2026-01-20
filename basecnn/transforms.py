from torchvision import transforms


def get_train_transform():
    """
    This function returns the transforms to apply to training images.
    Includes aggressive data augmentation for better generalization on small datasets.
    
    Returns:
        transform: torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transform():
    """
    This function returns the transforms to apply to test images.
    IMPORTANT: These should match the transforms you used during training
    (without data augmentation).
    
    Returns:
        transform: torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
