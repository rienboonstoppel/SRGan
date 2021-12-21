from torchvision import transforms
from generator import *
from discriminator import *
from feature_extractor import *
from trainer import *
from utils import *
from torch.utils.data import DataLoader

std = 0.3548
output_dir = 'runs/test_long'
start_epoch = 1
total_epochs = 100
lambda_edge = 1
lambda_pixel = 1
lambda_adv = 1
lambda_content = 1

transform = transforms.Compose([
    ToTensor(),
    Normalize(std=std),
])

tra_set = ImagePairDataset('training', patients_frac=.5, transform=transform)
val_set = ImagePairDataset('validation', patients_frac=.5, transform=transform)

print('Length of training set: \t{}\nLength of validation set: \t{}'.format(len(tra_set), len(val_set)))

batch_size = 8
n_cpu = 2
tra_dataloader = DataLoader(
    tra_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

val_dataloader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1).to(device)
discriminator = Discriminator(input_shape=(1, 224, 224)).to(device)
feature_extractor = FeatureExtractor().to(device)

trainer = Trainer(
    generator=generator,
    discriminator=discriminator,
    feature_extractor=feature_extractor,
    training_loader=tra_dataloader,
    validation_loader=val_dataloader,
    start_epoch=start_epoch,
    n_epochs=total_epochs,
    output_dir=output_dir,
    batch_size=batch_size,
    std=std,
    lambda_edge=lambda_edge,
    lambda_pixel=lambda_pixel,
    lambda_adv=lambda_adv,
    lambda_content=lambda_content,
)

metrics = trainer.fit()

