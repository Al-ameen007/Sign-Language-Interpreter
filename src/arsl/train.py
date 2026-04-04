import argparse
import os
import pickle

import torch

from arsl.data import create_data_loaders, save_dataset
from arsl.engine import save_model, train_model
from arsl.model import create_model
from arsl.utils import get_device, load_data_csv, plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Train Sign Language GCN Model")

    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="Saved")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--lr-decay-step", type=int, default=40)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_data_path = os.path.join(args.save_dir, "train_data.pth")
    test_data_path = os.path.join(args.save_dir, "test_data.pth")
    encoder_path = os.path.join(args.save_dir, "label_encoder.pkl")

    preprocessed_exists = (
        os.path.exists(train_data_path) and os.path.exists(test_data_path) and os.path.exists(encoder_path)
    )

    use_existing = False

    if preprocessed_exists and not args.force_preprocess:
        print("\n" + "=" * 60)
        print("FOUND EXISTING PREPROCESSED DATA!")
        print("=" * 60)
        print(f"  {train_data_path}")
        print(f"  {test_data_path}")
        print(f"  {encoder_path}")
        print("\nThis will save time by skipping video preprocessing.")

        while True:
            user_input = input("\nUse existing data? (y/n): ").lower().strip()
            if user_input in ["y", "n"]:
                break
            print("Please enter 'y' or 'n'")

        use_existing = user_input == "y"

    if use_existing:
        print("\n" + "=" * 60)
        print("LOADING EXISTING PREPROCESSED DATA")
        print("=" * 60)

        try:
            train_data = torch.load(train_data_path, weights_only=False)
            test_data = torch.load(test_data_path, weights_only=False)

            with open(encoder_path, "rb") as f:
                label_encoder = pickle.load(f)

            print(f"  Training samples: {len(train_data)}")
            print(f"  Test samples: {len(test_data)}")
            print(f"  Number of classes: {len(label_encoder.classes_)}")

        except Exception as e:
            print(f"\nError loading preprocessed data: {e}")
            print("Will reprocess videos instead...")
            use_existing = False

    if not use_existing:
        print("\n" + "=" * 60)
        print("PREPROCESSING VIDEOS")
        print("=" * 60)

        if preprocessed_exists and args.force_preprocess:
            print("Force preprocessing enabled - will overwrite existing data")

        print("\nLoading data from CSV files...")
        x_train, y_train, x_test, y_test = load_data_csv(args.train_csv, args.test_csv)

        print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

        print("\nPreparing training dataset with MediaPipe...")
        from preprocessing import process_videos_parallel

        train_processed = process_videos_parallel(x_train.tolist(), y_train.tolist(), max_workers=args.max_workers)

        print("\nPreparing test dataset with MediaPipe...")
        test_processed = process_videos_parallel(x_test.tolist(), y_test.tolist(), max_workers=args.max_workers)

        print("\nCreating graph datasets...")
        from data import create_graph_data_list
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        label_encoder.fit(y_train.tolist() + y_test.tolist())

        x_train_list = [item[0] for item in train_processed]
        adj_train_list = [item[1] for item in train_processed]
        y_train_list = [item[2] for item in train_processed]

        x_test_list = [item[0] for item in test_processed]
        adj_test_list = [item[1] for item in test_processed]
        y_test_list = [item[2] for item in test_processed]

        train_data = create_graph_data_list(x_train_list, adj_train_list, y_train_list, label_encoder)
        test_data = create_graph_data_list(x_test_list, adj_test_list, y_test_list, label_encoder)

        print("\nSaving datasets...")
        save_dataset(train_data, label_encoder, train_data_path, test_data_path, encoder_path, is_train=True)
        save_dataset(test_data, label_encoder, train_data_path, test_data_path, encoder_path, is_train=False)

        print("Preprocessing complete and data saved!")

    num_classes = len(label_encoder.classes_)

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Number of classes: {num_classes}")

    print("\nCreating data loaders...")
    train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size=args.batch_size)

    print("Initializing model with spatial attention...")
    model = create_model(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    device = get_device()
    print(f"Training on device: {device}")

    print("\nStarting training with paper's hyperparameters...")
    print(f"  Initial LR: {args.learning_rate}")
    print(f"  LR decay every {args.lr_decay_step} epochs by factor {args.lr_decay_gamma}")

    history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
    )

    model_path = os.path.join(args.save_dir, "gcn_model.pth")
    save_model(model, model_path)
    print(f"\nModel saved to {model_path}")

    print("\nFinal Results:")
    print(f"Train Accuracy: {history['train_accuracies'][-1]:.4f}")
    print(f"Test Accuracy: {history['test_accuracies'][-1]:.4f}")

    if not args.no_plot:
        print("\nPlotting training history...")
        plot_training_history(history)


if __name__ == "__main__":
    main()
