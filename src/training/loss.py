import numpy as np

# class BinaryCrossEntropy:
#     def forward(self, y_true, y_pred):
#         """
#         y_true: (batch_size, 1) or (batch_size,)
#         y_pred: (batch_size, 1) probabilities
#         """

#         # защита от log(0)
#         eps = 1e-15
#         y_pred = np.clip(y_pred, eps, 1 - eps)

#         loss = -np.mean(
#             y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)     
#         )

#         return loss
    


class CrossEntropyLoss:

    def forward(self, y_true, y_pred):

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(
            np.sum(y_true * np.log(y_pred), axis=1)
        )

        return loss    