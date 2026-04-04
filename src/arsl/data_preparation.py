import os
import shutil

import pandas as pd


def create_combined_folder(base_path, combined_folder_name):
    combined_path = os.path.join(base_path, combined_folder_name)
    if not os.path.exists(combined_path):
        os.makedirs(combined_path)
    return combined_path


def copy_contents(src, dst):
    for root, _dirs, files in os.walk(src):
        relative_path = os.path.relpath(root, src)
        target_dir = os.path.join(dst, relative_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)


def combine_folders(base_path, folders_to_combine, combined_folder_name="combined"):
    combined_path = create_combined_folder(base_path, combined_folder_name)

    for folder in folders_to_combine:
        src_path = os.path.join(base_path, folder)
        if os.path.exists(src_path):
            copy_contents(src_path, combined_path)
        else:
            print(f"Warning: {src_path} does not exist")

    print(f"Combined contents of {folders_to_combine} into {combined_path}")
    return combined_path


def verify_combined_folder(combined_path, folder_type="train/test"):
    """Verify the combined folder has video files"""
    print(f"\n--- Verifying {folder_type} folder ---")

    if not os.path.exists(combined_path):
        print(f"❌ ERROR: {combined_path} does not exist!")
        return False

    # Count videos and folders
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_count = 0
    folder_count = 0
    videos_by_folder = {}

    for root, dirs, files in os.walk(combined_path):
        if root == combined_path:
            folder_count = len(dirs)

        folder_name = os.path.basename(root)
        video_files = [f for f in files if f.lower().endswith(video_extensions)]

        if video_files:
            videos_by_folder[folder_name] = len(video_files)
            video_count += len(video_files)

    # Print results
    print(f"  📁 Total class folders: {folder_count}")
    print(f"  🎥 Total video files: {video_count}")

    if video_count > 0:
        print("  📊 Sample folders:")
        for folder, count in list(videos_by_folder.items())[:5]:
            print(f"     • {folder}: {count} videos")

    if video_count == 0:
        print(f"  ❌ ERROR: No video files found in {combined_path}")
        print("     Check that your source folders contain video files!")
        return False

    print("  ✓ Verification passed!")
    return True


def remove_leading_zeros_from_folders(directory_path):
    renamed_count = 0

    # Get all items in directory
    items = os.listdir(directory_path)

    for item in items:
        item_path = os.path.join(directory_path, item)

        if not os.path.isdir(item_path):
            continue

        # If this is a range folder (e.g., "0001-0070"), process its subfolders
        if "-" in item:
            print(f"  Processing range folder: {item}")
            subfolders = os.listdir(item_path)
            for subfolder in subfolders:
                old_subfolder_path = os.path.join(item_path, subfolder)

                if not os.path.isdir(old_subfolder_path):
                    continue

                # Remove leading zeros from subfolder name
                if subfolder.startswith("0") and len(subfolder) == 4:
                    new_subfolder_name = str(int(subfolder))
                    new_subfolder_path = os.path.join(item_path, new_subfolder_name)

                    if not os.path.exists(new_subfolder_path):
                        os.rename(old_subfolder_path, new_subfolder_path)
                        renamed_count += 1

        # Also handle regular folders (if any)
        elif item.startswith("0") and len(item) == 4:
            new_folder_name = str(int(item))
            new_folder_path = os.path.join(directory_path, new_folder_name)

            if not os.path.exists(new_folder_path):
                os.rename(item_path, new_folder_path)
                renamed_count += 1

    print(f"  Renamed {renamed_count} folders (removed leading zeros)")


def assign_labels(files_path, df, label_column="Sign-Arabic"):
    data_list = []

    folders = os.listdir(files_path)
    folders = [f for f in folders if os.path.isdir(os.path.join(files_path, f))]

    print(f"Found {len(folders)} top-level folders in {files_path}")

    for folder in folders:
        folder_path = os.path.join(files_path, folder)

        if "-" in folder:
            print(f"\nProcessing range folder: {folder}")

            subfolders = os.listdir(folder_path)
            subfolders = [f for f in subfolders if os.path.isdir(os.path.join(folder_path, f))]

            for subfolder in subfolders:
                try:
                    folder_sign_number = int(subfolder.lstrip("0")) if subfolder != "0000" else 0
                except ValueError:
                    print(f"  Skipping non-numeric subfolder: {subfolder}")
                    continue

                if folder_sign_number in df["SignID"].values:
                    row = df[df["SignID"] == folder_sign_number].iloc[0]
                    subfolder_path = os.path.join(folder_path, subfolder)
                    label = row[label_column]

                    video_count = 0
                    for file_name in os.listdir(subfolder_path):
                        if file_name.endswith((".mp4", ".avi", ".mov", ".mkv")):
                            file_path = os.path.join(subfolder_path, file_name)
                            data_list.append({"file_path": file_path, "Label": label, "SignID": folder_sign_number})
                            video_count += 1

                    if video_count > 0:
                        print(f"  • SignID {folder_sign_number} ({label}): {video_count} videos")
                else:
                    print(f"  Warning: SignID {folder_sign_number} not found in labels")

        else:
            try:
                folder_sign_number = int(folder)
            except ValueError:
                print(f"Skipping non-numeric folder: {folder}")
                continue

            if folder_sign_number in df["SignID"].values:
                row = df[df["SignID"] == folder_sign_number].iloc[0]
                label = row[label_column]

                video_count = 0
                for file_name in os.listdir(folder_path):
                    if file_name.endswith((".mp4", ".avi", ".mov", ".mkv")):
                        file_path = os.path.join(folder_path, file_name)
                        data_list.append({"file_path": file_path, "Label": label, "SignID": folder_sign_number})
                        video_count += 1

                if video_count > 0:
                    print(f"  • SignID {folder_sign_number} ({label}): {video_count} videos")
                elif video_count == 0:
                    print(f"  Warning: No videos found in {folder_path}")
            else:
                print(f"  Warning: SignID {folder_sign_number} not found in labels")

    if len(data_list) == 0:
        raise ValueError("No data found! Check your folder structure and video files.")

    data = pd.DataFrame(data_list)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n✓ Created dataset with {len(data)} videos and {data['Label'].nunique()} unique labels")

    return data


def prepare_karsl_simple(
    base_path, labels_file, output_csv_dir="csvs", force_recombine=False, skip_train=False, skip_test=False
):
    print("=" * 60)
    print("KARSL-502 Data Preparation (Simple Version)")
    print("=" * 60)

    df_labels = pd.read_excel(labels_file)
    print(f"\nLoaded {len(df_labels)} sign labels from Excel")

    os.makedirs(output_csv_dir, exist_ok=True)

    # Define paths
    combined_train = os.path.join(base_path, "combined_train")
    combined_test = os.path.join(base_path, "combined_test")
    train_csv_path = os.path.join(output_csv_dir, "train.csv")
    test_csv_path = os.path.join(output_csv_dir, "test.csv")

    # Check for existing train CSV
    if os.path.exists(train_csv_path) and not force_recombine and not skip_train:
        print(f"\n✓ Found existing train CSV: {train_csv_path}")
        user_input = input("  Skip train processing? (y/n): ").lower()
        skip_train = user_input == "y"

    # Check for existing test CSV
    if os.path.exists(test_csv_path) and not force_recombine and not skip_test:
        print(f"\n✓ Found existing test CSV: {test_csv_path}")
        user_input = input("  Skip test processing? (y/n): ").lower()
        skip_test = user_input == "y"

    # === TRAIN PROCESSING ===
    if not skip_train:
        print("\n" + "=" * 60)
        print("PROCESSING TRAIN DATA")
        print("=" * 60)

        # Check if train folder already combined
        train_needs_combining = True
        if os.path.exists(combined_train) and not force_recombine:
            print("\n✓ Found existing combined train folder")
            if verify_combined_folder(combined_train, "train"):
                print("  Skipping train combine step...")
                train_needs_combining = False
            else:
                print("  ⚠️ Verification failed, will recombine...")
                shutil.rmtree(combined_train)

        # Combine train folders if needed
        if train_needs_combining:
            print("\n1. Combining train folders from 01, 02, 03...")
            train_folders = [
                os.path.join(base_path, "01", "train"),
                os.path.join(base_path, "02", "train"),
                os.path.join(base_path, "03", "train"),
            ]
            combined_train = combine_folders(base_path, train_folders, "combined_train")

            # Verify after combining
            if not verify_combined_folder(combined_train, "train"):
                raise ValueError("Train folder verification failed after combining!")

        print("\n2. Removing leading zeros from train folders...")
        remove_leading_zeros_from_folders(combined_train)

        print("\n3. Creating train CSV...")
        train_data = assign_labels(combined_train, df_labels)
        train_data.to_csv(train_csv_path, index=False)
        print(f"   ✓ Saved to {train_csv_path}")
    else:
        print("\n⏭️  Skipping train processing (already exists)")
        train_data = pd.read_csv(train_csv_path)

    # === TEST PROCESSING ===
    if not skip_test:
        print("\n" + "=" * 60)
        print("PROCESSING TEST DATA")
        print("=" * 60)

        # Check if test folder already combined
        test_needs_combining = True
        if os.path.exists(combined_test) and not force_recombine:
            print("\n✓ Found existing combined test folder")
            if verify_combined_folder(combined_test, "test"):
                print("  Skipping test combine step...")
                test_needs_combining = False
            else:
                print("  ⚠️ Verification failed, will recombine...")
                shutil.rmtree(combined_test)

        # Combine test folders if needed
        if test_needs_combining:
            print("\n4. Combining test folders from 01, 02, 03...")
            test_folders = [
                os.path.join(base_path, "01", "test"),
                os.path.join(base_path, "02", "test"),
                os.path.join(base_path, "03", "test"),
            ]
            combined_test = combine_folders(base_path, test_folders, "combined_test")

            # Verify after combining
            if not verify_combined_folder(combined_test, "test"):
                raise ValueError("Test folder verification failed after combining!")

        print("\n5. Removing leading zeros from test folders...")
        remove_leading_zeros_from_folders(combined_test)

        print("\n6. Creating test CSV...")
        test_data = assign_labels(combined_test, df_labels)
        test_data.to_csv(test_csv_path, index=False)
        print(f"   ✓ Saved to {test_csv_path}")
    else:
        print("\n⏭️  Skipping test processing (already exists)")
        test_data = pd.read_csv(test_csv_path)

    # === FINAL SUMMARY ===
    print("\n" + "=" * 60)
    print("✓ PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nDataset Statistics:")
    print(f"  Train: {len(train_data)} videos, {train_data['Label'].nunique()} classes")
    print(f"  Test:  {len(test_data)} videos, {test_data['Label'].nunique()} classes")
    print("\nOutput files:")
    print(f"  📄 {train_csv_path}")
    print(f"  📄 {test_csv_path}")
    print("\nNext step:")
    print(f"  python train_paper.py --train-csv {train_csv_path} --test-csv {test_csv_path}")

    return train_csv_path, test_csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare KARSL-502 dataset (Simple Version)")
    parser.add_argument("--base-path", type=str, required=True, help="Path to folder containing 01, 02, 03 folders")
    parser.add_argument("--labels-file", type=str, required=True, help="Path to KARSL-502_Labels.xlsx")
    parser.add_argument("--csv-dir", type=str, default="csvs", help="Output directory for CSV files")
    parser.add_argument(
        "--force-recombine", action="store_true", help="Force recombining folders even if they already exist"
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip train processing (use existing train.csv)")
    parser.add_argument("--skip-test", action="store_true", help="Skip test processing (use existing test.csv)")

    args = parser.parse_args()

    prepare_karsl_simple(
        args.base_path,
        args.labels_file,
        args.csv_dir,
        force_recombine=args.force_recombine,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
    )
