def remove_empty_lines(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.strip():
                outfile.write(line)

remove_empty_lines('../assets/raw/dandelion.csv', '../assets/formatted/dandelion_clean.csv')
remove_empty_lines('../assets/raw/grass.csv', '../assets/formatted/grass_clean.csv')
