import numpy as np

class CMAC:

    def __init__(self, n_inputs, resolution, receptive, n_outputs, epochs, lr):
        self.n_inputs = n_inputs
        # resolution = 50
        self.resolution = resolution
        self.receptive = receptive
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.lr = lr
    
    def random_init_layer2(self):
        # random initilization 
        rng_x = np.random.default_rng(seed=42)
        rng_y = np.random.default_rng(seed=12)
        rec_x = rng_x.integers(self.receptive, size= self.receptive)
        rec_y = rng_y.integers(self.receptive, size= self.receptive)

        rand_init = np.zeros((self.resolution, self.resolution))

        for x in range(self.resolution):
            for y in range(self.resolution):
                for i in range(self.receptive):
                    # give the wieghts 1
                    if ((x % self.receptive) == rec_x[i]) and ((y % self.receptive) == rec_y[i]):
                        rand_init[x][y] = 1
        return rand_init

    
    def calculate_output(self, weights, X, output):
        
        # Layer 1
        layer1 = np.zeros((self.resolution, self.n_inputs))
        ID_x = X*self.resolution - 1
        for i in range(len(X)):
            layer1[int(ID_x[i]-self.receptive/2):int(ID_x[i]+self.receptive/2),i] = 1
        
        # Layer 2
        # with the size res x res
        layer2 = np.zeros((self.resolution,self.resolution))
        for i in range(self.resolution):
            for j in range(self.resolution):
                if output[i, j] == 1:
                    if (layer1.T[1, j] == 1) and (layer1[i, 0] == 1):
                        layer2[i, j] = 1

        # Layer 3
        # output layer
        layer3 = np.zeros(self.n_outputs)
        for i in range(self.n_outputs):
            layer3[i] = np.sum(weights[i,:,:]*layer2)

        return layer2, layer3
    
    def calculate_loss(self, y_pred, y_true):
        diff = y_true - y_pred
        loss = np.mean(np.power(diff, 2))
        return loss
    
    def update_weights(self, layer2, weights, y_pred, y_true):
        for n in range(len(y_pred)):
            diff = y_true[n] - y_pred[n]
            weights[n, :, :] = weights[n, :, :] + self.lr * layer2 / self.receptive * diff
        return weights

