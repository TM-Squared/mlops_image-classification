import csv


def generate_insert_sql(csv_file: str, output_sql_file: str, label:str):
    """
    Génère un fichier SQL d'insertion à partir d'un fichier CSV contenant des URLs.
    :param csv_file:
    :param output_sql_file:
    :param label:
    :return:
    """
    with open(csv_file, 'r') as infile, open(output_sql_file, 'w') as outfile:
        csv_reader = csv.reader(infile)

        # Commence le fichier SQL
        outfile.write("USE plants;\n")
        outfile.write("SET NAMES utf8mb4;\n")
        outfile.write("START TRANSACTION;\n")

        for row in csv_reader:
            if row:
                url_source = row[0]  # URL dans la première colonne
                insert_sql = f"INSERT INTO plants_data (url_source, label) VALUES ('{url_source}', '{label}');\n"
                # Écrire l'instruction dans le fichier SQL
                outfile.write(insert_sql)

        # Fin de la transaction SQL
        outfile.write("COMMIT;\n")


csv_file = '../assets/formatted/dandelion_clean.csv'
output_sql_file = '../scripts/insert_dandelion.sql'
generate_insert_sql(csv_file, output_sql_file, 'dandelion')

csv_file = '../assets/formatted/grass_clean.csv'
output_sql_file = '../scripts/inserts_grass.sql'
generate_insert_sql(csv_file, output_sql_file, 'grass')

print("SQL files generated.")
