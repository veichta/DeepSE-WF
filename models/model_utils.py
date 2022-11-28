import logging
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.data_utils import get_dataloader

REPRESENTATIONS = ["timing", "directional"]


def get_model(args, is_timing):
    """Get the model for the given representation.

    Args:
        args: Arguments passed to the script
        is_timing: True if the model is for timing

    Raises:
        NotImplementedError: If the model is not implemented.

    Returns:
        model: The model
    """
    if args.model == "df":
        from models.df import DF

        model = DF(include_classifier=True, args=args)

    elif args.model == "awf_cnn":
        from models.awf import AWF

        model = AWF(include_classifier=True, args=args)

    elif args.model == "var_cnn":
        from models.varcnn import VARCNN

        model = VARCNN(time=is_timing, include_classifier=True, args=args)

    elif args.model == "tf":
        from models.df import DF
        from models.triplet import TripletNetwork

        backbone = DF(include_classifier=False, args=args)

        model = TripletNetwork(embedding_net=backbone)

    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

    return model


def train_one_epoch(args, model, train_loader, optimizer, loss_fn, epoch):
    """Train the model for one epoch.

    Args:
        args: Arguments passed to the script
        model: Model to train
        train_loader: Data loader for training data
        optimizer: Optimizer
        loss: Loss function

    Returns:
        train_loss: Training loss
        train_acc: Training accuracy
    """
    model.to(args.device)
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    pbar = tqdm(total=len(train_loader), bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
    pbar.set_description(f"Epoch {epoch+1} / {args.epochs}")
    if args.model == "tf":
        if args.knn_measure == "cosine":
            from models.triplet import triplet_cosine_acc as acc_fn
        else:
            from models.triplet import triplet_l2_acc as acc_fn

        for i, triplets in enumerate(train_loader):
            anchor = triplets[0].to(args.device)
            positive = triplets[1].to(args.device)
            negative = triplets[2].to(args.device)

            optimizer.zero_grad()

            outputs = model(anchor, positive, negative)
            loss = loss_fn(outputs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc_fn(outputs)

            pbar.update(1)
            pbar.set_postfix(loss=train_loss / (i + 1), acc=train_acc / (i + 1))
    else:
        for i, (traces, targets) in enumerate(train_loader):
            traces = traces.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()

            pred = model(traces)

            loss = loss_fn(pred, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (pred.argmax(1) == targets).sum().item() / len(targets)

            pbar.update(1)
            pbar.set_postfix(loss=train_loss / (i + 1), acc=train_acc / (i + 1))

    pbar.close()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc


def validate(args, model, val_loader, loss_fn, epoch=None):
    """Validate the model.

    Args:
        args: Arguments passed to the script
        model: Model to validate
        val_loader: Data loader for validation data
        loss: Loss function

    Returns:
        val_loss: Validation loss
        val_acc: Validation accuracy
    """
    model.to(args.device)
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    pbar = tqdm(total=len(val_loader), bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
    if epoch is not None:
        pbar.set_description(f"Eval  {epoch+1} / {args.epochs}")
    else:
        pbar.set_description("Eval Test Set")

    with torch.no_grad():
        if args.model == "tf":
            if args.knn_measure == "cosine":
                from models.triplet import triplet_cosine_acc as acc_fn
            else:
                from models.triplet import triplet_l2_acc as acc_fn

            for i, triplets in enumerate(val_loader):
                anchor = triplets[0].to(args.device)
                positive = triplets[1].to(args.device)
                negative = triplets[2].to(args.device)

                outputs = model(anchor, positive, negative)
                loss_val = loss_fn(outputs)

                val_loss += loss_val.item()
                val_acc += acc_fn(outputs)

                pbar.update(1)
                pbar.set_postfix(val_loss=val_loss / (i + 1), val_acc=val_acc / (i + 1))
        else:
            for i, (traces, labels) in enumerate(val_loader):
                traces = traces.to(args.device)
                labels = labels.to(args.device)

                outputs = model(traces)
                loss_val = loss_fn(outputs, labels)
                val_loss += loss_val.item()
                val_acc += (outputs.argmax(1) == labels).sum().item() / len(labels)

                pbar.update(1)
                pbar.set_postfix(val_loss=val_loss / (i + 1), val_acc=val_acc / (i + 1))

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    return val_loss, val_acc


def get_embeddings(model, data_loader, args):
    """Get the embeddings for the given data loader.

    Args:
        model: Model to use
        data_loader: Data loader
        args: Arguments passed to the script

    Returns:
        embeddings: Embeddings
    """
    model.to(args.device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        if args.model == "tf":
            for triplets in data_loader:
                anchor = triplets[0].to(args.device)
                positive = triplets[1].to(args.device)
                negative = triplets[2].to(args.device)

                embedding = model(anchor, positive, negative)[0]
                embeddings.append(embedding.cpu().numpy())
        else:
            model.classifier = torch.nn.Identity()
            for traces, _ in data_loader:
                traces = traces.to(args.device)

                embedding = model(traces)
                embeddings.append(embedding.cpu().numpy())

    return np.concatenate(embeddings)


def train_models(data, args):
    """This function trains the attack for timeing as well as directional traces.

    Args:
        data: Dictionary containing train, test1 and test2
        args: Arguments passed to the script

    Raises:
        NotImplementedError: If the attack is not implemented.

    Returns:
        embeddings: Embeddings of the test data
    """
    embeddings = {}

    histories = {}
    for representation in REPRESENTATIONS:
        train_start = timer()

        is_timing = representation == "timing"

        # build model
        model = get_model(args, is_timing)

        # data loader
        x_train, x_val, y_train, y_val = train_test_split(
            data["x_train"],
            data["y_train"],
            test_size=0.1,
            stratify=data["y_train"],
            shuffle=True,
            random_state=42,
        )

        train_loader = get_dataloader(
            traces=x_train, labels=y_train, is_timing=is_timing, is_training=True, args=args
        )
        val_loader = get_dataloader(
            traces=x_val, labels=y_val, is_timing=is_timing, is_training=False, args=args
        )

        # optimizer
        optimizer = torch.optim.Adamax(
            model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0
        )
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        # loss function
        loss_fn = None
        if args.model == "tf":
            if args.knn_measure == "cosine":
                from models.triplet import triplet_cosine_loss

                loss_fn = triplet_cosine_loss
            else:
                from models.triplet import triplet_l2_loss

                loss_fn = triplet_l2_loss
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        current_history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_acc": [],
            "val_acc": [],
            "test_acc": [],
        }
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
            )
            val_loss, val_acc = validate(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                epoch=epoch,
                args=args,
            )

            current_history["train_loss"].append(train_loss)
            current_history["val_loss"].append(val_loss)
            current_history["train_acc"].append(train_acc)
            current_history["val_acc"].append(val_acc)

        train_end = timer()
        logging.info(f"\t{representation} model ({timedelta(seconds=train_end-train_start)})")

        # get embeddings
        test1_loader = get_dataloader(
            traces=data["x_test1"],
            labels=data["y_test1"],
            is_timing=is_timing,
            is_training=False,
            args=args,
        )
        test2_loader = get_dataloader(
            traces=data["x_test2"],
            labels=data["y_test2"],
            is_timing=is_timing,
            is_training=False,
            args=args,
        )

        test1_loss, test1_acc = validate(
            model=model,
            val_loader=test1_loader,
            loss_fn=loss_fn,
            args=args,
        )
        test2_loss, test2_acc = validate(
            model=model,
            val_loader=test2_loader,
            loss_fn=loss_fn,
            args=args,
        )

        current_history["test_loss"].append((test1_loss + test2_loss) / 2)
        current_history["test_acc"].append((test1_acc + test2_acc) / 2)

        histories[representation] = current_history

        embeddings[representation] = {
            "test1": get_embeddings(model=model, data_loader=test1_loader, args=args),
            "test2": get_embeddings(model=model, data_loader=test2_loader, args=args),
        }

    return embeddings, histories
