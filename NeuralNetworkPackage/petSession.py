import torch
from CNN import CNN
from Data import *





inference = False
multiOut = False
datasetNumber = 1
imgHeight = 256
imgWidth = 256
colorChannels = 3


if inference:

    datasetSize = 50


    (imgBatch, labelBatch) = genPetImageStack(datasetSize, img_height=imgHeight, img_width=imgWidth, multi_out=multiOut, save=False, dataset_number=datasetNumber, color_channels=colorChannels)
    # (imgBatch, labelBatch) = loadPetImageStack(multi_out=multiOut)


    cnn = CNN(pretrained=True, training=False, cnn_model_params=fetchCNNParametersFromFile(), mlp_model_params=fetchMLPParametersFromFile())

    predictionBatch = cnn.inference(imgBatch)

    printPetInferenceResults(dataset_size=datasetSize, img_batch=imgBatch, label_batch=labelBatch, prediction_batch=predictionBatch, color_channels=colorChannels)




else:

    # isConvLayer =          [True, False, True, False, True, False]
    # filterCounts =         [5, 5, 8, 8, 10, 10] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIOAL LAYER KERNEL
    # kernelShapes =         [(3, 3), (2, 2), (3, 3), (2, 2), (3, 3), (2, 2)]
    # kernelStrides =        [1, 2, 1, 2, 1, 2]
    # CNNactivationFunctions =  ["leakyReLU", "none", "leakyReLU", "none", "leakyReLU", "none"]
    #
    # neuronCounts = [64, 32, None]
    # MLPactivationFunctions = ["leakyReLU", "leakyReLU", "none"]



    # isConvLayer =            [True, False, True, False, True, False]
    # filterCounts =           [32, 32, 64, 64, 128, 128] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIOAL LAYER KERNEL
    # kernelShapes =           [(3, 3), (2, 2), (3, 3), (2, 2), (3, 3), (2, 2)]
    # kernelStrides =          [1, 2, 1, 2, 1, 2]
    # CNNactivationFunctions = ["leakyReLU", "none", "leakyReLU", "none", "leakyReLU", "none"]
    #
    # neuronCounts = [128, 64, None]
    # MLPactivationFunctions = ["leakyReLU", "leakyReLU", "none"]

    # isConvLayer =            [True, True, False, True, True, False, True, True, False]
    # filterCounts =           [16, 32, 32, 64, 128, 128, 256, 256, 256] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIOAL LAYER KERNEL
    # kernelShapes =           [(3, 3), (3, 3), (2, 2), (3, 3), (3, 3), (2, 2), (3, 3), (3, 3), (2, 2)]
    # kernelStrides =          [1, 1, 2, 1, 1, 2, 1, 1, 2]
    # CNNactivationFunctions = ["leakyReLU", "leakyReLU", "none", "leakyReLU", "leakyReLU", "none", "leakyReLU", "leakyReLU", "none"]



    # Architecture 3
    # isConvLayer =            [True, False, True, False, True, True, False]
    # filterCounts =           [16, 16, 64, 64, 128, 256, 256] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIOAL LAYER KERNEL
    # kernelShapes =           [(3, 3), (2, 2), (3, 3), (2, 2), (3, 3), (3, 3), (2, 2)]
    # kernelStrides =          [1, 2, 1, 2, 1, 1, 2]
    # CNNactivationFunctions = ["leakyReLU", "none", "leakyReLU", "none", "leakyReLU", "leakyReLU", "none"]
    #
    # neuronCounts = [1024, 512, 256, 128, 64, 64, 32, 32, 16, 8, 4, None]
    # MLPactivationFunctions = ["leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "none"]



    # isConvLayer =            [True, True, False, True, True, False, True, True, False, True, True, False, True, True, False]
    # filterCounts =           [8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIOAL LAYER KERNEL
    # kernelShapes =           [(3, 3), (3, 3), (2, 2), (3, 3), (3, 3), (2, 2), (3, 3), (3, 3), (2, 2), (3, 3), (3, 3), (2, 2), (3, 3), (3, 3), (2, 2)]
    # kernelStrides =          [1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1]
    # CNNactivationFunctions = ["leakyReLU", "leakyReLU", "none", "leakyReLU", "leakyReLU", "none", "leakyReLU", "leakyReLU", "none", "leakyReLU", "leakyReLU", "none", "leakyReLU", "leakyReLU", "none"]
    #
    # neuronCounts = [128, 64, 48, 24, 12, None]
    # MLPactivationFunctions = ["leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "none"]

    isConvLayer =          [True, False, True, False]
    filterCounts =         [2, 2, 4, 4] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIOAL LAYER KERNEL
    kernelShapes =         [(5, 5), (2, 2), (3, 3), (2, 2)]
    kernelStrides =        [1, 2, 1, 2]
    CNNactivationFunctions =  ["leakyReLU", "none", "sigmoid", "none"]

    neuronCounts =        [10, 1]
    MLPactivationFunctions = ["leakyReLU", "none"]



    neuronCounts[-1] = 2 if multiOut else 1
    CNNmodelConfig = {
        "is_conv_layer": isConvLayer,
        "filter_counts": filterCounts,
        "kernel_shapes": kernelShapes,
        "kernel_strides": kernelStrides,
        "CNN_activation_functions": CNNactivationFunctions
    }
    MLPmodelConfig = {
        "neuron_counts": neuronCounts,
        "MLP_activation_functions": MLPactivationFunctions
    }

    CNNHyperParameters = {
        "learn_rate": 0.01,
        "batch_size": 256,
        "loss_func": "BCEWithLogitsLoss",
        "reduction": "mean",
        "optimizer": "adam",
        "lambda_L2": None, #1e-6 
        "dropout_rate": None #0.2,
    }
    MLPHyperParameters = {
        "learn_rate": 0.01,
        "optimizer": "adam",
        "lambda_L2": None, #1e-6,
        "dropout_rate": None,
    }



    datasetSize = 5000

    (imgBatch, labelBatch) = genPetImageStack(datasetSize, img_height=imgHeight, img_width=imgWidth, multi_out=multiOut, save=False, dataset_number=datasetNumber, color_channels=colorChannels)
    # (imgBatch, labelBatch) = loadPetImageStack(multi_out=multiOut)

    cnn = CNN(pretrained=False, training=True, hyperparameters=CNNHyperParameters, mlp_hyperparameters=MLPHyperParameters, input_data_dim=(colorChannels, imgHeight, imgWidth), cnn_model_config=CNNmodelConfig, mlp_model_config=MLPmodelConfig)
    # cnn = CNN(pretrained=True, training=True, hyperparameters=CNNHyperParameters, mlp_hyperparameters=MLPHyperParameters, cnn_model_params=fetchCNNParametersFromFile(), mlp_model_params=fetchMLPParametersFromFile())

    (epochPlt, lossPlt) = cnn.train(imgBatch, labelBatch, epochs=500)

    plotTrainingResults(epochPlt, lossPlt)

