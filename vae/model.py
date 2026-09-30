import numpy as np

from optim import adam_init, adam_step


def Relu(x):
    return np.maximum(0, x)


class VAE:
    def __init__(self, X, lr = 0.001, epochs = 1000):
        self.lr = lr
        self.epochs = epochs

        self.original_data = X
        self.data_mean = np.mean(X, axis=0)
        self.data_std = np.std(X, axis=0)
        self.X = (X - self.data_mean) / self.data_std

        self.W1_enc = np.random.randn(8, 2) * np.sqrt(1 / 5)
        self.b1_enc = np.zeros((8, 1))
        self.W2_enc = np.random.randn(8, 8) * np.sqrt(1 / 8)
        self.b2_enc = np.zeros((8, 1))

        self.W_mean = np.random.randn(2, 8) * np.sqrt(1 / 5)
        self.b_mean = np.zeros((2, 1))
        self.W_log_var = np.random.randn(2, 8) * np.sqrt(1 / 5)
        self.b_log_var = np.zeros((2, 1))

        self.W1_dec = np.random.randn(8, 2) * np.sqrt(1 / 5)
        self.b1_dec = np.zeros((8, 1))
        self.W2_dec = np.random.randn(8, 8) * np.sqrt(1 / 8)
        self.b2_dec = np.zeros((8, 1))
        self.W_out = np.random.randn(2, 8) * np.sqrt(1 / 5)
        self.b_out = np.zeros((2, 1))

        self.ls = np.zeros((2, 1))

        self.params = {
            "W1_enc": self.W1_enc, "b1_enc": self.b1_enc,
            "W2_enc": self.W2_enc, "b2_enc": self.b2_enc,
            "W_mean": self.W_mean, "b_mean": self.b_mean,
            "W_log_var": self.W_log_var, "b_log_var": self.b_log_var,
            "W1_dec": self.W1_dec, "b1_dec": self.b1_dec,
            "W2_dec": self.W2_dec, "b2_dec": self.b2_dec,
            "W_out": self.W_out, "b_out": self.b_out,
            "ls" : self.ls
        }

        self.states = adam_init(self.params)


    def encode(self):
        self.enc1 = Relu(self.W1_enc @ self.X.T + self.b1_enc)
        self.enc2 = Relu(self.W2_enc @ self.enc1 + self.b2_enc)

        self.mean = self.W_mean @ self.enc2 + self.b_mean
        self.log_var = self.W_log_var @ self.enc2 + self.b_log_var
        self.eps = np.random.normal(size = self.mean.shape)

        return self.mean + np.exp(0.5 * self.log_var) * self.eps

    def decode(self, z):
        self.dec1 = Relu(self.W1_dec @ z + self.b1_dec)
        self.dec2 = Relu(self.W2_dec @ self.dec1 + self.b2_dec)
        out = self.W_out @ self.dec2 + self.b_out
        return out

    def forward(self):
        self.z = self.encode()
        self.out = self.decode(self.z)
        return self.out


    def loss(self):
        N = self.X.shape[0]
        inv = np.exp(-self.ls)
        r = self.out - self.X.T

        recon_loss = 0.5 * np.sum(r ** 2 * inv) / N + 0.5 * np.sum(self.ls)
        kl_loss = 0.5 * np.mean(
            np.sum(np.exp(self.log_var) + self.mean ** 2 - 1.0 - self.log_var, axis=0)
        )
        self.loss_val = recon_loss + kl_loss
        return self.loss_val


    def train(self, X):
        N = self.X.shape[0]
        t = 0

        for epoch in range(self.epochs):
            self.forward()
            z = self.mean + np.exp(0.5 * self.log_var) * self.eps

            inv = np.exp(-self.ls)
            r = self.out - self.X.T

            d_ls = 0.5 * (1.0 - inv * (np.sum(r ** 2, axis=1, keepdims=True) / N))

            d_out = (r * inv) / N
            dW_out = d_out @ self.dec2.T
            db_out = d_out.sum(1, keepdims=True)
            d_dec2 = (self.W_out.T @ d_out) * (self.dec2 > 0)
            dW2_dec = d_dec2 @ self.dec1.T
            db2_dec = d_dec2.sum(1, keepdims=True)
            d_dec1 = (self.W2_dec.T @ d_dec2) * (self.dec1 > 0)
            dW1_dec = d_dec1 @ z.T
            db1_dec = d_dec1.sum(1, keepdims=True)
            d_z = self.W1_dec.T @ d_dec1

            d_mean = d_z + (self.mean / N)
            d_log_var = 0.5 * d_z *(self.z - self.mean) + (np.exp(self.log_var) - 1) / (2 * N)

            dW_mean = d_mean @ self.enc2.T
            db_mean = d_mean.sum(1, keepdims=True)
            dW_log_var = d_log_var @ self.enc2.T
            db_log_var = d_log_var.sum(1, keepdims=True)
            d_enc2 = (self.W_mean.T @ d_mean + self.W_log_var.T @ d_log_var) * (self.enc2 > 0)

            dW2_enc = d_enc2 @ self.enc1.T
            db2_enc = d_enc2.sum(1, keepdims=True)
            d_enc1 = (self.W2_enc.T @ d_enc2) * (self.enc1 > 0)
            dW1_enc = d_enc1 @ self.X
            db1_enc = d_enc1.sum(1, keepdims=True)

            self.grads = {
                "W1_enc": dW1_enc, "b1_enc": db1_enc,
                "W2_enc": dW2_enc, "b2_enc": db2_enc,
                "W_mean": dW_mean, "b_mean": db_mean,
                "W_log_var": dW_log_var, "b_log_var": db_log_var,
                "W1_dec": dW1_dec, "b1_dec": db1_dec,
                "W2_dec": dW2_dec, "b2_dec": db2_dec,
                "W_out": dW_out, "b_out": db_out,
                "ls": d_ls
            }

            t += 1
            adam_step(self.params, self.grads, self.states, t, lr=self.lr)
            np.clip(self.ls, -8.0, 4.0, out=self.ls)

            if epoch % (self.epochs // 10) == 0 or epoch == self.epochs - 1:
                recon = 0.5 * np.sum(r ** 2 * inv) / N + 0.5 * np.sum(self.ls)
                kl = 0.5 * np.mean(np.sum(
                    np.exp(self.log_var) + self.mean ** 2 - 1.0 - self.log_var, axis=0))
                print(f"epoch {epoch:4d} | loss {recon + kl:.4f} | "
                      f"recon {recon:.4f} | kl {kl:.4f} | ls {self.ls.ravel()}")


    def generate(self, num_samples):
        z = np.random.normal(size=(2, num_samples))
        out = self.decode(z)
        sigma = np.exp(0.5 * self.ls)
        out = out + sigma * np.random.normal(size=out.shape)
        return out.T * self.data_std + self.data_mean
