import torch
import numpy as np
torch.set_printoptions(threshold=torch.inf)
import glob, os, re
import matplotlib.pyplot as plt, matplotlib.pylab as pylab
from mnistData import train_dataset, test_dataset
from datasets import load_dataset
import random
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader





def printPetInferenceResults(dataset_size, img_batch, label_batch, prediction_batch, color_channels):

    if dataset_size == 1:

        predicted_animal = "dog" if prediction_batch.item() >= 0 else "cat"
        animal_label = "dog" if label_batch.item() == 1 else "cat"


        print(f"prediction = {prediction_batch}")
        print(f"predicted animal = {predicted_animal}")

        print(f"truth: {label_batch}")
        print(f"truth label = {animal_label}")


        plt.imshow(img_batch.squeeze(), cmap='gray')
        plt.title(f'Prediction: {predicted_animal}, Label: {animal_label}')
        plt.show()



    else:

        predictions = ["dog" if pet >= 0 else "cat" for pet in prediction_batch[0].tolist()]
        labels = ["dog" if pet_label == 1 else "cat" for pet_label in label_batch.tolist()]
        print(prediction_batch)

        print(f"predictions = {predictions}")
        print(f"labels      = {labels}")

        num_correct = torch.sum(torch.abs(prediction_batch - label_batch) < 1)

        percent_correct = (num_correct/dataset_size)*100
        print(f"percent correct = {percent_correct.item()}%")


        for (img, pred, lbl) in zip(img_batch, predictions, labels):

            if color_channels == 1:
                plt.imshow(img.squeeze(), cmap='gray') # grey images

            elif color_channels == 3:
                img = np.transpose(img.numpy(), (1, 2, 0)) # color images
                plt.imshow(img)

            plt.title(f'Prediction: {pred}, Label: {lbl}')

            plt.show(block=False)
            plt.pause(1.5)
            plt.close()


def loadPetImageStack(multi_out):
    if multi_out:
        img_batch = torch.load("petdatatensors_multiout.pth")
        label_batch = torch.load("pettargettensors_multiout.pth")
    else:
        img_batch = torch.load("petdatatensors.pth")
        label_batch = torch.load("pettargettensors.pth")
    return img_batch, label_batch


def getPetImageTensor(transform, seed, dataset_number):

    if dataset_number == 1:
        image_path = f'dogs-vs-cats/train/{seed}.{random.randint(0, 12499)}.jpg'
        image = Image.open(image_path)

    elif dataset_number == 2:
        image_path = f'train_{seed}/{random.randint(0, 416)}.jpg'
        image = Image.open(image_path)

    else:
        seed = random.randint(0, 11740) if seed == "cat" else random.randint(11741, 23410)
        ds = load_dataset("microsoft/cats_vs_dogs")['train']
        image = ds[seed]["image"]

    # Load, resize, and transform image
    image_tensor = transform(image)

    return image_tensor


def genPetImageStack(dataset_size, img_height, img_width, multi_out, save, dataset_number, color_channels):

    # Create dataset transformation
    transformations = [
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ]

    # Preallocate tensors for the entire dataset
    data_tensors = torch.zeros((dataset_size, color_channels, img_height, img_width))  # Assuming RGB images
    target_tensors = torch.zeros(dataset_size) if not multi_out else torch.zeros(size=(2, dataset_size))

    for n in range(dataset_size):
        print(f"generating training data.... {(n/dataset_size)*100}%")
        # Generate random index and select seed

        # seed = "dog" if n % 2 == 0 else "cat"
        seed = "dog" if n % 2 == 0 else "cat"

        if color_channels == 1:
            transformations.append(transforms.Grayscale(num_output_channels=1))

        transform = transforms.Compose(transformations)
        image_tensor = getPetImageTensor(transform, seed, dataset_number)

        # Insert into preallocated tensor
        data_tensors[n] = image_tensor

        if not multi_out:
            target_tensors[n] = 1 if seed == "dog" else 0
        elif multi_out:
            target_tensors[0, n] = 1 if seed == "dog" else 0
            target_tensors[1, n] = 1 if seed == "cat" else 0


        # image_np = np.transpose(image_tensor.numpy(), (1, 2, 0))
        # plt.imshow(image_np)
        # plt.show()

    if save:
        if multi_out:
            torch.save(data_tensors, "petdatatensors_multiout.pth")
            torch.save(target_tensors, "pettargettensors_multiout.pth")
        else:
            torch.save(data_tensors, "petdatatensors.pth")
            torch.save(target_tensors, "pettargettensors.pth")


    return data_tensors, target_tensors





def printMNISTInferenceResults(dataset_size, img_batch, label_batch, prediction_batch):

    if dataset_size == 1:

        prediction = torch.argmax(prediction_batch)
        label = torch.argmax(label_batch)

        print(f"prediction:")
        print(prediction_batch)
        print(f"predicted number = {prediction}")
        print(f"truth: {label}")


        plt.imshow(img_batch.squeeze(), cmap='gray')
        plt.title(f'Prediction: {prediction}, Label: {label}')
        # plt.show()

        plt.show(block=False)



    else:

        predictions = torch.argmax(prediction_batch, dim=0)
        labels = torch.argmax(label_batch, dim=0)

        print(f"predictions = {predictions}")
        print(f"labels      = {labels}")

        num_correct = torch.sum(predictions == labels)

        percent_correct = (num_correct/dataset_size)*100
        print(f"percent correct = {percent_correct.item()}%")


        for (img, pred, lbl) in zip(img_batch, predictions, labels):

            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'Prediction: {pred}, Label: {lbl}')

            plt.show(block=False)
            plt.pause(1.5)
            plt.close()


def genOneHotEncodedMNISTStack(dataset_size, inference):

    data_tensors = []
    target_tensors = []

    dataset = test_dataset if inference else train_dataset

    max_index = len(dataset)
    rand_indicies = torch.randint(low=0, high=max_index, size=(dataset_size, ))

    for n in rand_indicies:

        n = n.item()

        sample, label = dataset[n]

        one_hot_encoded_label = torch.zeros(size=(10, ))
        one_hot_encoded_label[label] = 1

        data_tensors.append(sample)
        target_tensors.append(one_hot_encoded_label.T)


    data_batch = torch.stack(data_tensors, dim=0)
    target_batch = torch.stack(target_tensors, dim=1)

    return data_batch, target_batch


def plotTrainingResults(epoch_plt, loss_plt):

    epoch_plt = torch.tensor(epoch_plt)
    loss_plt = torch.tensor(loss_plt)
    print(f"mean loss unfiltered = {loss_plt.mean()}")

    loss_filter = loss_plt > loss_plt.mean()
    mask = torch.ones(loss_filter.size(0), dtype=torch.bool)
    mask[loss_filter] = False
    # Apply the mask
    epoch_plt = epoch_plt[mask]
    loss_plt = loss_plt[mask]
    print(f"mean loss filtered = {loss_plt.mean()}")


    plt.figure(1)
    marker_size = 1
    f = plt.scatter(epoch_plt[:], loss_plt[:], s=marker_size)
    plt.xlabel("epoch")
    plt.ylabel("loss")


    z = np.polyfit(epoch_plt, loss_plt, 5)
    p = np.poly1d(z)
    pylab.plot(epoch_plt, p(epoch_plt), "r--")


    plt.show()


def fetchMLPParametersFromFile():

    modelParams = {}

    # Define the directory and pattern
    directory = 'parametersMLP/' #os.getcwd()  # Replace with the directory path

    # Use glob to get all files matching the pattern
    weight_pattern = "layer_*_weights_*.pth"  # Pattern to match
    weight_files = glob.glob(os.path.join(directory, weight_pattern))
    weight_files.sort()

    bias_pattern = "layer_*_biases_*.pth"  # Pattern to match
    bias_files = glob.glob(os.path.join(directory, bias_pattern))
    bias_files.sort()


    for (w_file, b_file) in zip(weight_files, bias_files):

        # weights = torch.tensor(np.genfromtxt(w_file, delimiter=','))
        # biases = torch.tensor(np.genfromtxt(b_file,  delimiter=',')).reshape(-1, 1)

        weights = torch.load(w_file)
        biases = torch.load(b_file)

        regex_pattern = r"layer_(\d+)_weights_(.*?)\.pth"
        match = re.search(regex_pattern, w_file)

        index = match.group(1)
        activation = match.group(2)

        modelParams.update({f"Layer {index}": [weights, biases, activation, index] })

    return modelParams


def fetchCNNParametersFromFile():

    modelParams = {}

    # Define the directory and pattern
    directory = 'parametersCNN/' #os.getcwd()  # Replace with the directory path


    # Use glob to get all files matching the pattern
    kernel_pattern = "cnn_layer_*_kernels_*_*_*.pth"  # Pattern to match
    kernel_files = glob.glob(os.path.join(directory, kernel_pattern))
    kernel_files.sort()

    bias_pattern = "cnn_layer_*_biases_*_*_*.pth"  # Pattern to match
    bias_files = glob.glob(os.path.join(directory, bias_pattern))
    bias_files.sort()


    for (k_file, b_file) in zip(kernel_files, bias_files):

        # print(k_file, b_file)

        kernels = torch.load(k_file)
        biases = torch.load(b_file)

        # regex_pattern = r"cnn_layer_(\d+)_kernels_(.*?)_(.*?)_(\d+)\.pth"
        regex_pattern = r"cnn_layer_(\d+)_kernels_(\w+)_([\w]+)_(\d+)\.pth"

        match = re.search(regex_pattern, k_file)

        index = match.group(1)
        activation = match.group(2)
        is_conv = match.group(3)
        stride = match.group(4)

        modelParams.update({f"CNN Layer {index}": [is_conv, kernels, biases, activation, stride, index] })

    return modelParams





if __name__ == "__main__":

    # imgHeight = 5
    # imgWidth = 5
    #
    # d, t = genPetImageStack(2, imgHeight, imgWidth, False, save=False, dataset_number=2, color_channels=3)
    #
    # print(d)
    # # print(t)
    #
    # for d_, t_ in zip(d, t):
    #     # print(d_)
    #     # print(t_)
    #     print(f"mean = {torch.mean(d_)}")
    #     image_np = np.transpose(d_.numpy(), (1, 2, 0))
    #     plt.imshow(image_np, cmap="gray")
    #     # plt.imshow(image_np)
    #     plt.show()







    n = fetchMLPParametersFromFile()
    print(n)





