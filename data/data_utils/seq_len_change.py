import csv

freq = 100

with open('../file_markers_dodh/train_file_markers.csv', 'r') as f:
    reader = csv.reader(f)
    with open('train_file_markers_'+str(freq)+'.csv', 'w') as w:
        writer = csv.writer(w)
        for row in reader:
            row[3] = 30 * freq
            writer.writerow(row)

with open('../file_markers_dodh/test_file_markers.csv', 'r') as f:
    reader = csv.reader(f)
    with open('test_file_markers_'+str(freq)+'.csv', 'w') as w:
        writer = csv.writer(w)
        for row in reader:
            row[3] = 30 * freq
            writer.writerow(row)

with open('../file_markers_dodh/val_file_markers.csv', 'r') as f:
    reader = csv.reader(f)
    with open('val_file_markers_'+str(freq)+'.csv', 'w') as w:
        writer = csv.writer(w)
        for row in reader:
            row[3] = 30 * freq
            writer.writerow(row)