import logging

from torch import nn


class DF(nn.Module):
    def __init__(self, include_classifier, args):
        """Initialize the df model architecture.

        Args:
            seq_len: Length of the input sequence.
            include_classifier: Whether to include the classifier or not
            args: Arguments

        Returns:
            model: Pytorch model which implements the DF attack neural network
        """
        super(DF, self).__init__()

        logging.debug(
            f"Using DF model with {args.embedding_size} embedding "
            + f"size (classifier: {include_classifier})"
        )

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(num_features=32),
            nn.ELU(alpha=1.0),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(num_features=32),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),
            nn.Dropout(p=0.2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(num_features=64),
            nn.ELU(alpha=1.0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(num_features=64),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),
            nn.Dropout(p=0.2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(num_features=128),
            nn.ELU(alpha=1.0),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(num_features=128),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),
            nn.Dropout(p=0.2),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(num_features=256),
            nn.ELU(alpha=1.0),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(num_features=256),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),
            nn.Dropout(p=0.2),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=512, out_features=args.embedding_size),
        )

        self.classifier = None

        if include_classifier:
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(args.embedding_size),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=args.embedding_size, out_features=args.n_websites),
            )

    def forward(self, x):
        """Do a forward pass of the model.

        Args:
            x: Input data.

        Returns:
            Output of the model.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.embedding(x)

        if self.classifier is not None:
            x = self.classifier(x)

        return x
