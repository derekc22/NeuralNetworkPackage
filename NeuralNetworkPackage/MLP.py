import torch
from Network import Network
import numpy as np
from FullyConnectedLayer import FullyConnectedLayer



class MLP(Network):

    def __init__(self, pretrained, training, **kwargs):

        if not pretrained:
            mlp_model_config = kwargs.get("mlp_model_config")
            self.checkConfig(model_config=mlp_model_config)
            self.layers = MLP.buildLayers(mlp_model_config=mlp_model_config, input_feature_count=kwargs.get("input_feature_count"))
        else:
            self.layers = MLP.loadLayers(mlp_model_params=kwargs.get("mlp_model_params"))

        self.num_layers = len(self.layers)

        super().__init__(model_type="MLP", training=training, kwargs=kwargs)









    @staticmethod
    def loadLayers(mlp_model_params):
        layers = [FullyConnectedLayer(pretrained=True, pretrained_weights=weights, pretrained_biases=biases, nonlinearity=nonlinearity, index=index) for (weights, biases, nonlinearity, index) in mlp_model_params.values()]
        return layers


    @staticmethod
    def buildLayers(mlp_model_config, input_feature_count):

        neuron_counts = mlp_model_config.get("neuron_counts")
        activation_functions = mlp_model_config.get("MLP_activation_functions")
        # neuron_counts, activation_functions = mlp_model_config.values()

        neuron_counts.insert(0, input_feature_count)

        # NOT including the input layer
        num_layers = len(neuron_counts)
        #output_layer_size = neuron_counts[-1]

        layers = [FullyConnectedLayer(pretrained=False, input_count=neuron_counts[i], neuron_count=neuron_counts[i+1], nonlinearity=activation_functions[i], index=i+2) for i in range(num_layers-1)]

        return layers


    def saveParameters(self):
        for layer in self.layers:
            layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
            torch.save(layer.weights, f"parametersMLP/layer_{layer.index}_weights_{layer.nonlinearity}.pth")
            torch.save(layer.biases, f"parametersMLP/layer_{layer.index}_biases_{layer.nonlinearity}.pth")





    def forward(self, curr_input, training):
        for layer in self.layers:

            curr_input = layer.feed(curr_input)

            if training and self.dropout_rate and layer != self.layers[-1]:
                curr_input = self.dropout(curr_input)

        return curr_input




    def dropout(self, curr_input):
        # print("FC DROPOUT")

        drop_count = int(self.dropout_rate * curr_input.numel())
        dropout_row_indicies = torch.randint(low=0, high=curr_input.size(dim=0), size=(drop_count,))
        dropout_col_indicies = torch.randint(low=0, high=curr_input.size(dim=1), size=(drop_count,))

        curr_input[dropout_row_indicies, dropout_col_indicies] = 0

        return curr_input






    # def backprop(self, loss):
    #
    #     self.zerograd()
    #
    #     loss.backward()
    #
    #     with torch.no_grad():
    #
    #         if not self.optimizer:
    #             for layer in self.layers:
    #                 self.update(layer)
    #         else:
    #             self.t += 1
    #             for layer in self.layers:
    #                 self.optimizerUpdate(layer)



