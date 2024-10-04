import numpy as np
from Layer import Layer
import torch
import torch.nn as nn



class ConvolutionalLayer(Layer):


    def __init__(self, pretrained, is_conv_layer, **kwargs):

        self.is_conv_layer = is_conv_layer
        self.padding = 0
        self.nonlinearity = kwargs.get("nonlinearity")
        self.index = int(kwargs.get("index"))
        self.kernel_stride = int(kwargs.get("kernel_stride"))


        if not pretrained:
            self.filter_count = kwargs.get("filter_count")
            self.kernel_height, self.kernel_width = kwargs.get("kernel_shape")

            # Random Initilization
            # dim=0 is set to 1, allowing the kernel to expand to match the batch size of the input image
            self.kernels = torch.rand(size=(1, self.filter_count, self.kernel_height, self.kernel_width), dtype=torch.float64) if is_conv_layer else torch.empty(size=(1, self.filter_count, self.kernel_height, self.kernel_width), dtype=torch.float64)
            self.biases = torch.rand(size=(1, self.filter_count, 1), dtype=torch.float64) if is_conv_layer else None

            # # He Initialization
            # input_count = self.filter_count * self.kernel_height * self.kernel_width
            # stddev = np.sqrt(2 / input_count)
            # self.kernels = torch.normal(0, stddev, size=(1, self.filter_count, self.kernel_height, self.kernel_width), dtype=torch.float64) if is_conv_layer else torch.empty(size=(1, self.filter_count, self.kernel_height, self.kernel_width), dtype=torch.float64)

        else:
            self.kernels = kwargs.get("pretrained_kernels")
            self.biases = kwargs.get("pretrained_biases")
            self.filter_count, self.kernel_height, self.kernel_width = self.kernels.size()[-3:]


        if is_conv_layer:
            self.kernels.requires_grad_()
            self.biases.requires_grad_()




    def __repr__(self):

        return (f"__________________________________________\n"
                f"CNN Layer {self.index}\nKernels:\n{self.kernels}\nKernel Size: {self.kernels.size()}\nBiases:\n{self.biases}\nBias Size: {self.biases.size() if self.biases is not None else None}\nActivation: {self.nonlinearity}\n"
                f"__________________________________________")





    def traverse(self, imgs, func):
        # print("traverse")



        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(dim=1)

        # img_batch_size = imgs.size(dim=0)
        # img_channel_count = imgs.size(dim=1)
        # img_height = imgs.size(dim=2)
        # img_width = imgs.size(dim=3)

        img_batch_size, img_channel_count, img_height, img_width = imgs.size()

        img_slices_stack = imgs.unfold(dimension=2, size=self.kernel_height, step=self.kernel_stride).unfold(dimension=3, size=self.kernel_width, step=self.kernel_stride).reshape(img_batch_size, img_channel_count, -1, self.kernel_height, self.kernel_width)
        num_slices = img_slices_stack.size(dim=2)
        # print(f"num_slices = {num_slices}")
        # print(f"img_slice_stack = {img_slices_stack.size()}")


        result = func(img_slices_stack)


        feature_map_rows = int(((img_height - self.kernel_height + 2*self.padding)/self.kernel_stride) + 1)
        feature_map_cols = int(((img_width - self.kernel_width + 2*self.padding)/self.kernel_stride) + 1)
        feature_map = result.reshape(img_batch_size, self.filter_count, feature_map_rows, feature_map_cols)
        # print(f"feature map = {feature_map.size()}")



        if self.is_conv_layer:

            ####### TESTING THIS ############################
            # feature_map = feature_map/(torch.max(feature_map)) # best during inference
            # feature_map = feature_map/(torch.norm(feature_map))

            if img_batch_size > 1:
                bn2 = nn.BatchNorm2d(num_features=self.filter_count, dtype=torch.float64)
                feature_map = bn2(feature_map)


            # normalizer = torch.max(torch.max(feature_map, dim=2).values, dim=2).values
            # normalizer = normalizer.reshape(img_batch_size, -1, 1, 1)  # Reshape the normalizer to enable broadcasting
            # feature_map = feature_map / normalizer
            ####### TESTING THIS ############################



            feature_map = self.activate(feature_map, self.nonlinearity)



        return feature_map




    def convolve(self, x):
        # print("convolve")

        def f(img_slices_stack):

            result = torch.einsum('bcshw,bfhw->bfshw', img_slices_stack, self.kernels)
            result = torch.sum(result, dim=(3, 4)) + self.biases

            return result

        return self.traverse(x, func=f)



    def maxpool(self, x):
        # print("maxpool")

        def f(img_slices_stack):

            result = img_slices_stack.max(dim=3)[0].max(dim=3)[0]

            return result

        return self.traverse(x, func=f)









