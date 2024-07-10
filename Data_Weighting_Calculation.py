import os

KNOWN_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces'
NON_FACES_DIR = 'C:\\DARAM-ai-V2\\non_faces'
OTHER_FACES_DIR = 'C:\\DARAM-ai-V2\\knows_faces\\Other'



def count_files_in_directory(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])


def calculate_weights(known_faces_dir, non_faces_dir, other_faces_dir):
    known_faces_counts = {d: count_files_in_directory(os.path.join(known_faces_dir, d)) for d in
                          os.listdir(known_faces_dir) if d.isdigit()}
    non_faces_count = count_files_in_directory(non_faces_dir)
    other_faces_count = count_files_in_directory(other_faces_dir)
    total_files = sum(known_faces_counts.values()) + non_faces_count + other_faces_count
    known_faces_weights = {label: count / total_files for label, count in known_faces_counts.items()}
    non_faces_weight = non_faces_count / total_files
    other_faces_weight = other_faces_count / total_files
    return known_faces_weights, non_faces_weight, other_faces_weight


def print_weights(known_faces_weights, non_faces_weight, other_faces_weight):
    print("Known Faces Weights:")
    for label, weight in known_faces_weights.items():
        print(f"Label {label}: {weight:.4f}")
    print(f"Non Faces Weight: {non_faces_weight:.4f}")
    print(f"Other Faces Weight: {other_faces_weight:.4f}")


if __name__ == "__main__":
    known_faces_weights, non_faces_weight, other_faces_weight = calculate_weights(KNOWN_FACES_DIR, NON_FACES_DIR,
                                                                                  OTHER_FACES_DIR)
    print_weights(known_faces_weights, non_faces_weight, other_faces_weight)
