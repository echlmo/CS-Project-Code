import csv
import json
import requests

# Get a dictionary of the words -> human readable
words_csv = csv.reader(open("words.txt","r"),delimiter="\t")
words = {x[0]:x[1] for x in words_csv}

# Get the children of the 5 classes
base_url = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid="
ids = [("wagon", "n02814533"),
       ("cat", "n02121808"),
       ("dog", "n02084071"),
       ("flower", "n11669921"),
       ("mushroom", "n07734744")]

results = dict()

# Get the subset ID's of each
for item in ids:

    # Make the request
    request = requests.get(base_url + item[1] + "&full=1")

    # Split the list (exclude original tag) and get human readable labels
    children = request.text.split("\r\n")[1:-1]
    children_results = [{"id":child[1:],"label":words[child[1:]]} for child in children]

    # Add to the results
    results[item[0]] = children_results

# Save as a JSON file
json.dump(results,open("children.json","w"))
