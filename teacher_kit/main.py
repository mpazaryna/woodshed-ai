from teacher_kit.utils.json_util import read_json_file


def main():
    # Specify the path to your JSON file
    file_path = "teacher_kit/data/asana.json"

    # Read the JSON file
    data = read_json_file(file_path)

    # Check if the data is a list and has at least one element
    if isinstance(data, list) and data:
        # Print the first element
        print("First element of the JSON file:", data[0])
    else:
        print("The JSON file is empty or not a list.")


if __name__ == "__main__":
    main()
