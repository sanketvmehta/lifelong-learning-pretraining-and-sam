from functools import partial, partialmethod
import torchvision
import torch
from torchvision.models.resnet import (
    BasicBlock,
    _resnet,
    conv1x1,
    load_state_dict_from_url,
    ResNet as BaseResNet,
    model_urls,
)


class RNet(BaseResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        dropout=0.0,
    ):
        self._dropout = dropout
        super().__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                dropout=self._dropout,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    dropout=self._dropout,
                )
            )

        return torch.nn.Sequential(*layers)


class ResNet(torch.nn.Module):
    """
    Resnet18 (pretrained on Imagenet)
    """

    def __init__(self, total_classes=1000, layers=18, dropout=0.0, norm_layer=None):
        super(ResNet, self).__init__()
        if layers == 18:
            # resnet = torchvision.models.resnet18(num_classes=total_classes)
            resnet = _resnet(
                "resnet18",
                ResNetBlock,
                [2, 2, 2, 2],
                False,
                True,
                num_classes=total_classes,
                dropout=dropout,
                norm_layer=norm_layer
            )
        elif layers == 34:
            resnet = torchvision.models.resnet34(num_classes=total_classes)
        elif layers == 50:
            resnet = torchvision.models.resnet50(num_classes=total_classes)
        else:
            raise ValueError("not a recognized ResNet")
        for param in resnet.parameters():
            param.requires_grad = True
        self.net = resnet

    def forward(self, x):
        return self.net(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs) -> ResNet:
    model = RNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


class ResNetBlock(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
        dropout: float = 0,
    ) -> None:
        super().__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )
        self._dropout = dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout(out)

        return out
