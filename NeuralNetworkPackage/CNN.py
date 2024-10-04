import torch
from ConvolutionalLayer import ConvolutionalLayer
from Network import Network
from MLP import MLP
torch.set_printoptions(threshold=torch.inf)



class CNN(Network):

    def __init__(self, pretrained, training, **kwargs):

        if not pretrained:
            cnn_model_config = kwargs.get("cnn_model_config")
            self.checkConfig(model_config=cnn_model_config)
            self.layers = CNN.buildLayers(cnn_model_config=cnn_model_config)

            MLP_input_feature_count = self.calcMLPInputSize(kwargs.get("input_data_dim"))
            self.MLP = MLP(pretrained=False, training=training, input_feature_count=MLP_input_feature_count, mlp_model_config=kwargs.get("mlp_model_config"), hyperparameters=kwargs.get("mlp_hyperparameters"))
        else:
            self.layers = CNN.loadLayers(kwargs.get("cnn_model_params"))
            self.MLP = MLP(pretrained=True, training=training, mlp_model_params=kwargs.get("mlp_model_params"), hyperparameters=kwargs.get("mlp_hyperparameters"))


        self.num_layers = len(self.layers)

        super().__init__(model_type="CNN", training=training, kwargs=kwargs)





    @staticmethod
    def loadLayers(cnn_model_params):

        layers = [ConvolutionalLayer(pretrained=True, is_conv_layer=(is_conv_layer == "True"), pretrained_kernels=kernels, pretrained_biases=biases, nonlinearity=nonlinearity, kernel_stride=stride, index=index) for (is_conv_layer, kernels, biases, nonlinearity, stride, index) in cnn_model_params.values()]

        return layers


    @staticmethod
    def buildLayers(cnn_model_config):

        is_conv_layer = cnn_model_config.get("is_conv_layer")
        filter_counts = cnn_model_config.get("filter_counts")
        kernel_shapes = cnn_model_config.get("kernel_shapes")
        kernel_strides = cnn_model_config.get("kernel_strides")
        activation_functions = cnn_model_config.get("CNN_activation_functions")
        num_layers = len(is_conv_layer)

        layers = [ConvolutionalLayer(pretrained=False, is_conv_layer=is_conv_layer[i], filter_count=filter_counts[i], kernel_shape=kernel_shapes[i], kernel_stride=kernel_strides[i], nonlinearity=activation_functions[i], index=i+1) for i in range(num_layers)]

        return layers



    def saveParameters(self):
        for layer in self.layers:
            layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
            torch.save(layer.kernels, f"parametersCNN/cnn_layer_{layer.index}_kernels_{layer.nonlinearity}_{layer.is_conv_layer}_{layer.kernel_stride}.pth")
            torch.save(layer.biases, f"parametersCNN/cnn_layer_{layer.index}_biases_{layer.nonlinearity}_{layer.is_conv_layer}_{layer.kernel_stride}.pth")

        self.MLP.saveParameters()



    def calcMLPInputSize(self, input_data_dim):

        print("calculating MLP input size.......")

        input_data_dim = (1, ) + input_data_dim

        dummy_data = torch.empty(size=input_data_dim)
        dummy_MLP_input = self.forward(dummy_data, training=True, dummy=True)
        dummy_MLP_input_feature_count = dummy_MLP_input.size(dim=0)

        return dummy_MLP_input_feature_count






    def forward(self, curr_input, training, dummy=False):  # maybe recursion would work for this?

        # if not dummy:
        #     print("og image = ")
        #     if self.batch_size == 1:
        #         plt.imshow(curr_input.squeeze(dim=0).squeeze(dim=0))
        #         plt.show()

        for layer in self.layers:
            if layer.is_conv_layer:

                curr_input = layer.convolve(curr_input)

                # if self.dropout_rate:
                #     curr_input = self.spatialDropout(curr_input)

                # if training and not dummy and self.dropout_rate: #layer != self.layers[-1]:
                #     curr_input = layer.convolve(curr_input, dropout_rate=self.dropout_rate)
                # else:
                #     curr_input = layer.convolve(curr_input)


            else:
                curr_input = layer.maxpool(curr_input)

            # if not dummy:
            #     print("forward")
            #     print(f"mean img = {torch.mean(curr_input)}")
            #     print(f"mean kernel = {torch.mean(layer.kernels)}")
            #     print(f"kernel grad = {layer.kernels.grad}")
            #     print("************************************************************")
            #     print(curr_input)


        curr_input_batch_size = curr_input.size(dim=0)
        flattened_feature_map = curr_input.view(curr_input_batch_size, -1).to(torch.float64).T


        """flattened_feature_map = curr_input.view(curr_input_batch_size, -1).to(torch.float64).T""" 
        """flattened_feature_map = curr_input.reshape(-1, curr_input_batch_size).to(torch.float64)""" 
        """flattened_feature_map = curr_input.view(-1, curr_input_batch_size).to(torch.float64)""" 


        if dummy:
            return flattened_feature_map
        else:
            return self.MLP.forward(flattened_feature_map, training=training)




    # def spatialDropout(self, curr_input):
    #
    #     drop_count = int(self.dropout_rate * curr_input.size(dim=1))
    #
    #     dropout_indicies = torch.randint(low=0, high=curr_input.size(dim=1), size=(drop_count, ))
    #
    #     curr_input[:, dropout_indicies, :, :] = 0
    #
    #     # print(f"curr_input_dim = {curr_input.size()}. dim 1 = {curr_input.size(dim=1)}")
    #     # print(dropout_indicies)
    #     # print(curr_input)
    #
    #     return curr_input







    def backprop(self, loss):

        self.zerograd()
        self.MLP.zerograd()


        loss.backward()

        with torch.no_grad():

            if not self.optimizer:
                self.update()
            else:
                self.t += 1
                self.optimizerUpdate()


            if not self.MLP.optimizer:
                self.MLP.update()
            else:
                self.MLP.t += 1
                self.MLP.optimizerUpdate()

