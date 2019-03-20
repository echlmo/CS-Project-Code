import json
import csv


"""(OLD) Get top-1 results from top-5 results JSON into JSON file"""
def get_top1_results(filepath, results_file):

    with open(filepath) as f:
        data = json.load(f)

    results = dict()

    for img in data:
        results[img] = data[img]["0"]

    json.dump(results,open(results_file,"w"),indent=4,separators=(',',': '))

    return results


"""Read JSON of (top-5) results from experiment into CSV containing top-1 predictions
Grades predictions for 'hits' i.e. successful predictions as 1, else 0."""
def res_to_csv(filepath, results_file):

    columns = ['Image','Class ID','Class Name','Confidence','Hits']

    with open(filepath) as f:
        data = json.load(f)

    results = list()

    for key in list(data.keys()):

        dicdata = dict()

        dicdata["Image"] = key
        dicdata["Class ID"] = data[key]["0"]["class_id"]
        dicdata["Class Name"] = data[key]["0"]["class_name"]
        dicdata["Confidence"] = data[key]["0"]["confidence"]

        # Record hit/miss based on the original class
        dicdata["Classification Accuracy"] = classify_pred(key, data[key]["0"]["class_id"])

        results.append(dicdata)

    try:
        with open(results_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for data in results:
                writer.writerow(data)
    except IOError:
        print("I/O Error")


"""Classifies prediction as hit=1 or miss=0 according to prediction class
A 'hit' counts as a class which either matches the original image's true class
or is a subclass of the original image true class."""
def classify_pred(key, classname):

    # Open file containing data for the prediction class and its child classes
    with open("./children.json") as children:
        childlist = json.load(children)

    # Gets true class name from the parent file of the image
    c = key.split("/")[-2]

    if c == "cars":
        if classname in [childlist["wagon"][i]['id'] for i in range(len(childlist["wagon"]))]:
            result = 1
        else: result = 0
    if c == "cats":
        if classname in [childlist["cat"][i]['id'] for i in range(len(childlist["cat"]))]:
            result = 1
        else: result = 0
    if c == "dogs":
        if classname in [childlist["dog"][i]['id'] for i in range(len(childlist["dog"]))]:
            result = 1
        else: result = 0
    if c == "flowers":
        if classname in [childlist["flower"][i]['id'] for i in range(len(childlist["flower"]))]:
            result = 1
        else: result = 0
    if c == "mushrooms":
        if classname in [childlist["mushroom"][i]['id'] for i in range(len(childlist["mushroom"]))]:
            result = 1
        else: result = 0

    return result