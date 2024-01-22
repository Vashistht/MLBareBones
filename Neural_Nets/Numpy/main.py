"""Main script for the solution."""

import numpy as np
import pandas as pd
import argparse

import npnn


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument("--opt", help="optimizer", default="SGD")
    p.add_argument(
        "--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument(
        "--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument(
        "--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
    p.add_argument(
        "--test_dataset", help="Dataset file (test set)",
        default="mnist_test.npz")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()


if __name__ == '__main__':
    args = _get_args()
    X, y = npnn.load_mnist(args.dataset)
    print(X.shape)
    # Create dataset (see npnn/dataset.py)
    # train and val split
    indices = np.arange(X.shape[0]) 
    np.random.shuffle(indices)
    print(indices)
    train_i, val_i  = indices[:50000], indices[50000:]
    X_train, X_val = X[train_i], X[val_i]
    y_train, y_val = y[train_i], y[val_i]
    train_dataset = npnn.Dataset(X_train, y_train, batch_size = 32)
    val_dataset = npnn.Dataset(X_val, y_val, batch_size = 32)

    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    # Create model (see npnn/model.py)
    # Train for args.epochs
    # model = None
    
    # Create a NN with 3 Dense layers with 256, 64, and 10 units
    # Use ELU activation for the first two layers with alpha=0.9
    # Use SoftmaxCrossEntropy after the last one
    # user define args.opt
    if args.opt == "SGD":
        optimizer = npnn.SGD(learning_rate=args.lr)
    elif args.opt == "Adam":
        optimizer = npnn.Adam(learning_rate=args.lr)
    else:
        print("No optimizer specified, using SGD")
        optimizer = npnn.SGD(learning_rate=args.lr)
    
    loss = npnn.SoftmaxCrossEntropy()
    
    layers = [npnn.Flatten(), 
              npnn.Dense(28*28, 256), 
              npnn.ELU(alpha=0.9), 
              npnn.Dense(256, 64), 
              npnn.ELU(alpha=0.9), 
              npnn.Dense(64, 10)]    
    
    # putting in the model
    model = npnn.Sequential(layers, loss=loss, optimizer=optimizer)
    
    # train and val
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    for epoch in range(args.epochs):
        print("Epoch: {}".format(epoch))
        # train
        mean_loss, mean_acc = model.train(train_dataset)
        train_loss.append(mean_loss)
        train_acc.append(mean_acc)
        
        # val
        mean_loss, mean_acc = model.test(val_dataset)
        val_loss.append(mean_loss)
        val_acc.append(mean_acc)
        
        print("Train loss: {:.4f}, Train acc: {:.4f}".format(mean_loss, mean_acc))
        print("Val loss: {:.4f}, Val acc: {:.4f}".format(mean_loss, mean_acc))
    
    # add to data frame
    stats = pd.DataFrame()
    # stats[['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc']] = pd.DataFrame([[train_loss, train_acc, val_loss, val_acc]])
    stats ['Train Loss'] = train_loss
    stats ['Train Acc'] = train_acc
    stats ['Val Loss'] = val_loss
    stats ['Val Acc'] = val_acc

    # Save statistics to file.
    # We recommend that you save your results to a file, then plot them
    # separately, though you can also place your plotting code here.
    if args.save_stats:
        stats.to_csv("data/{}_{}.csv".format(args.opt, args.lr))

    # Save predictions.
    if args.save_pred:
        X_test, _ = npnn.load_mnist("mnist_test.npz")
        y_pred = np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
        np.save("mnist_test_pred.npy", y_pred)
