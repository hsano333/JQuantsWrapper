import torch
import torch.nn as nn
import torch.nn.functional as F

D_in = 10
H = 4
D_out = 1

# def images_to_probs(net, images):
#     '''
#     Generates predictions and corresponding probabilities from a trained
#     network and a list of images
#     '''
#     output = net(images)
#     # convert output probabilities to predicted class
#     _, preds_tensor = torch.max(output, 1)
#     preds = np.squeeze(preds_tensor.numpy())
#     return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
#
# def plot_classes_preds(net, images, labels):
#     preds, probs = images_to_probs(net, images)
#     # plot the images in the batch, along with predicted and true labels
#     fig = plt.figure(figsize=(12, 48))
#     for idx in np.arange(4):
#         ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
#         matplotlib_imshow(images[idx], one_channel=True)
#         ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
#             classes[preds[idx]],
#             probs[idx] * 100.0,
#             classes[labels[idx]]),
#                     color=("green" if preds[idx]==labels[idx].item() else "red"))
#     return fig


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU(inplace=False)

        self.container = nn.Sequential(
            torch.nn.Linear(D_in, H),
            self.relu,
            torch.nn.Linear(H, 2),
            self.relu,
            nn.Dropout(p=0.5),
            torch.nn.Linear(2, D_out),
        )

    def compute_batch_loss(
        self, ndx, data, label, y_pred_prob, batch_size, metrics, writer
    ):
        loss = nn.MSELoss()(y_pred_prob, label)
        if ndx % 10 == 1:
            writer.add_scalar("training loss", loss, ndx)
            # writer.add_figure(
            #     "predictions vs actual",
            #     plot_classes_preds(),
            #     global_step=ndx * batch_size,
            # )

        return loss

    def forward(self, x):
        result = self.container(x)
        return result
