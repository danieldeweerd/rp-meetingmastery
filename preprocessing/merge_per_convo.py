import os
import glob

# Directory path
dr = 'data/memo-corpus/vtt/sessions'

for i in range(1, 16):
    if i == 12:
        continue

    # Search pattern
    search_pattern = dr + os.sep + '*group{}*'.format(i)

    # Output file name
    output_file = 'data/memo-corpus/vtt/merged/group{}.tsv'.format(i)

    # Initialize an empty list to store file contents
    merged_contents = []

    # Find files matching the search pattern
    files = glob.glob(search_pattern)

    # Read and merge contents of matching files
    for file_path in files:
        with open(file_path, 'r') as file:
            merged_contents.append(file.read())

    # Write the merged contents to the output file
    with open(output_file, 'w') as output:
        output.write('\n'.join(merged_contents))

    # Print a message indicating the merge is complete
