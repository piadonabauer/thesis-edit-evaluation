import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_o_key, img_e_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images_original = df[img_o_key].tolist()
        self.images_edit = df[img_e_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images_original = self.transforms(Image.open(str(self.images_original[idx])))
        images_edit = self.transforms(Image.open(str(self.images_edit[idx])))
        #texts = self.tokenize([str(self.captions[idx])])[0]
        texts = self.tokenize(["An image edit of " + str(self.captions[idx])])[0] # CHANGED
        return images_original, images_edit, texts
    
def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_o_key=args.csv_img_o_key,
        img_e_key=args.csv_img_e_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, criterion, optimizer, epochs=5):
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, text, label in train_loader:
                # Concatenate features and move to device
                images = images.to(device)
                text = text.to(device)
                label = label.to(device).float()  # Ensure labels are float for BCELoss
                
                # Forward pass
                optimizer.zero_grad()
                output = self.forward(torch.cat((images, text), dim=1))
                loss = criterion(output.squeeze(), label)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader)}")

    def evaluate_model(self, val_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, text, label in val_loader:
                images = images.to(device)
                text = text.to(device)
                label = label.to(device).float()
                
                output = self.forward(torch.cat((images, text), dim=1))
                predicted = (output.squeeze() > 0.5).float()  # Binary prediction threshold
                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')

def main(args):
    # Model, loss, optimizer
    input_dim = 1024  # Example input dimension
    hidden_dim = 512  # Example hidden layer size
    model = MLPClassifier(input_dim, hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loader_info = get_csv_dataset(args, preprocess_fn, is_train=True, tokenizer=tokenizer)
    val_loader_info = get_csv_dataset(args, preprocess_fn, is_train=False, tokenizer=tokenizer)


    if args.mode == 'train':
        model.train_model(train_loader, criterion, optimizer, epochs=args.epochs)
    elif args.mode == 'test':
        model.evaluate_model(val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test the MLP classifier.")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode: train or test")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for data loading")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for training")

    parser.add_argument('--model', type=str, default='RN101', help='Type of model to load (default: RN101)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    
    args = parser.parse_args()

    main(args)

    parser.add_argument('--model', type=str, default='RN101', help='Type of model to load (default: RN101)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--name', type=str, required=True, help='Name of file to save the correlation df')

    #parser.add_argument('--image_original_path', type=str, required=True, help='Path to the original image')
    #parser.add_argument('--image_edit_path', type=str, required=True, help='Path to the edited image')
    #parser.add_argument('--instruction', type=str, required=True, help='Path to the instruction')

    args = parser.parse_args()
    main(args.model, args.checkpoint, args.name)#, args.image_original_path, args.image_edit_path)#, args.instruction) 