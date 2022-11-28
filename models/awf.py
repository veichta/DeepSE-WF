import logging

from torch import nn


class AWF(nn.Module):
    def __init__(self, include_classifier, args):
        """Initialize the awf model architecture.

        Args:
            include_classifier: Whether to include the classifier or not
            args: Arguments

        Returns:
            model: Pytorch model which implements the AWF attack neural network
        """
        super(AWF, self).__init__()

        logging.debug(
            f"Using AWF model with {args.embedding_size} embedding "
            + f"size (classifier: {include_classifier})"
        )

        self.conv_block1 = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        # calculate the output size of the conv layers
        n_features = (args.feature_length - 4) // 4
        n_features = (n_features - 4) // 4

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * n_features, out_features=args.embedding_size),
            nn.ReLU(),
        )

        self.classifier = None
        if include_classifier:
            self.classifier = nn.Sequential(
                nn.Dropout(p=args.dropout),
                nn.Linear(in_features=args.embedding_size, out_features=args.n_websites),
            )

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input data

        Returns:
            Output of the model
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.embedding(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x
