from CNN import CNN
from Data import fetchMLPParametersFromFile, fetchCNNParametersFromFile, plotTrainingResults, genOneHotEncodedMNISTStack, printMNISTInferenceResults
import torch





inference = True

if inference:

    datasetSize = 100

    (imgBatch, labelBatch) = genOneHotEncodedMNISTStack(datasetSize, inference=inference)

    cnn = CNN(pretrained=True, training=False, cnn_model_params=fetchCNNParametersFromFile(), mlp_model_params=fetchMLPParametersFromFile())
    # for layer in cnn.layers:
    #     print(layer)

    predictionBatch = cnn.inference(imgBatch)

    printMNISTInferenceResults(dataset_size=datasetSize, img_batch=imgBatch, label_batch=labelBatch, prediction_batch=predictionBatch)




else:

    isConvLayer =          [True, False, True, False]
    filterCounts =         [2, 2, 4, 4] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIOAL LAYER KERNEL
    kernelShapes =         [(5, 5), (2, 2), (3, 3), (2, 2)]
    kernelStrides =        [1, 2, 1, 2]
    CNNactivationFunctions =  ["leakyReLU", "none", "sigmoid", "none"]

    neuronCounts =        [10]
    MLPactivationFunctions = ["softmax"]

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
        "batch_size": 64,
        "loss_func": "CCELoss",
        "reduction": "mean",
        "optimizer": "adam",
        "lambda_L2": None, #1e-6
        "dropout_rate": None
    }
    MLPHyperParameters = {
        "learn_rate": 0.01,
        "optimizer": "adam",
        "lambda_L2": None, #1e-6
        "dropout_rate": None
    }



    datasetSize = 50_000

    (imgBatch, labelBatch) = genOneHotEncodedMNISTStack(datasetSize, inference=inference)

    cnn = CNN(pretrained=False, training=True, hyperparameters=CNNHyperParameters, mlp_hyperparameters=MLPHyperParameters, input_data_dim=(1, 28, 28), cnn_model_config=CNNmodelConfig, mlp_model_config=MLPmodelConfig)
    # cnn = CNN(pretrained=True, cnn_hyperparameters=CNNHyperParameters, mlp_hyperparameters=MLPHyperParameters, cnn_model_params=fetchCNNParametersFromFile(), mlp_model_params=fetchMLPParametersFromFile())

    (epochPlt, lossPlt) = cnn.train(imgBatch, labelBatch, epochs=1000)

    plotTrainingResults(epochPlt, lossPlt)



