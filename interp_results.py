import json
import csv

def get_top_results(filepath):
    with open(filepath) as f:
        data = json.load(f)

    results = dict()

    for img in data:
        results[img] = data[img]["0"]

    json.dump(results,open("./topresults.json","w"),indent=4,separators=(',',': '))

    return results

def res_to_csv(filepath):
    columns = ['Image','Class ID','Class Name','Confidence']

    with open(filepath) as f:
        data = json.load(f)

    results = list()

    for key in list(data.keys()):
        dicdata = dict()
        dicdata["Image"] = key
        dicdata["Class ID"] = data[key]["0"]["class_id"]
        dicdata["Class Name"] = data[key]["0"]["class_name"]
        dicdata["Confidence"] = data[key]["0"]["confidence"]
        results.append(dicdata)

    csv_file = "drresults.csv"
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for data in results:
                writer.writerow(data)
    except IOError:
        print("I/O Error")
