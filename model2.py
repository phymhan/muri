import torch.nn as nn
import torch


class C3D2(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, num_classes=2, arch=1, comb=1, fc_dim=4096):
        super(C3D2, self).__init__()

        self.arch = arch
        self.comb = comb

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, fc_dim)
        self.fc7 = nn.Linear(fc_dim, fc_dim)
        self.fc8 = nn.Linear(fc_dim, int(num_classes))

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def part1(self, x):
        h = self.relu(self.conv1(x))
        h = self.bn1(h)
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.bn2(h)
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.bn3(h)
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.bn4(h)
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.bn5(h)
        h = self.pool5(h)

        h = torch.mean(h, dim=2)

        h = h.view(-1, 8192)
        if self.arch == 1:
            return h

        h = self.fc6(h)
        if self.arch == 2:
            return h
        h = self.relu(h)
        h = self.dropout(h)

        h = self.fc7(h)
        if self.arch == 3:
            return h
        h = self.relu(h)
        h = self.dropout(h)

        logits = self.fc8(h)
        if self.arch == 4:
            return logits
        return logits

    def part2(self, h):
        logits = None

        if self.arch == 1:
            h = self.fc6(h)
            h = self.relu(h)
            h = self.dropout(h)
            h = self.fc7(h)
            h = self.relu(h)
            h = self.dropout(h)
            logits = self.fc8(h)

        if self.arch == 2:
            h = self.relu(h)
            h = self.dropout(h)
            h = self.fc7(h)
            h = self.relu(h)
            h = self.dropout(h)
            logits = self.fc8(h)

        if self.arch == 3:
            h = self.relu(h)
            h = self.dropout(h)
            logits = self.fc8(h)

        if self.arch == 4:
            logits = h

        return logits

    def forward(self, x1, x2):
        h = None

        h1 = self.part1(x1)
        h2 = self.part1(x2)

        if self.comb == 1:
            h = h1 - h2
        elif self.comb == 2:
            h = h1 + h2

        logits = self.part2(h)
        return logits

    def __call__(self, x):
        return nn.Softmax()(self.part2(self.part1(x)))

    def load_pretrained_weights(self, model_path):
        """Initialiaze network."""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",
        }

        p_dict = torch.load(model_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""