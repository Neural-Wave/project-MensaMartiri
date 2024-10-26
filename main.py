import utils
from dataloader import *


def run(): 
    # Assuming you have your CustomDataModule set up
    data_module = CustomDataModule(csv_file=config.EXPORT1, batch_size=32)

    # Call setup to initialize the datasets
    data_module.setup()

    # Get the train DataLoader
    train_loader = data_module.train_dataloader()

    # Visualize the first 4 images and their labels
    utils.visualize_images(train_loader, num_images=4)

if __name__ == "__main__":
    run()


