import argparse
import json
import matplotlib.pyplot as plt

# object properties: latency (float), numberOfSamples (int), predictionType (string: SAMPLES, TEST_SET, SAMPLED_TEST_SET), 
#                    modelName (string)

parser = argparse.ArgumentParser()
parser.add_argument("file_name", nargs='?', type=str, default="measurements.json", help="Name of a file to be analyzed.")
parser.add_argument("-p", "--phone", type=str, default="", help="Name of a phone, on which the tests were performed.")
args = parser.parse_args()

with open(args.file_name, "r") as file:
    data = json.load(file)

# remove duplicate measurements
data = [dict(t) for t in {tuple(d.items()) for d in data}]

with open(args.file_name, "w") as file:
    json.dump(data, file, indent=2)

samples = []
test_set = []
sampled_test_set = []
for measurement in data:
    if measurement["predictionType"] == "SAMPLES":
        samples.append({ "latency": measurement["latency"], "modelName": measurement["modelName"], "numberOfSamples": measurement["numberOfSamples"] })
    elif measurement["predictionType"] == "TEST_SET":
        test_set.append({ "latency": measurement["latency"], "modelName": measurement["modelName"] })
    elif measurement["predictionType"] == "SAMPLED_TEST_SET":
        sampled_test_set.append({ "latency": measurement["latency"], "modelName": measurement["modelName"] })

# order by latency
test_set.sort(key=lambda x: x["latency"])
sampled_test_set.sort(key=lambda x: x["latency"])
samples.sort(key=lambda x: x["latency"])

TITLE = True

# scatter plot results for test set measurements
plt.figure(figsize=(10, 4))
if TITLE:
    plt.title("Test Set Measurements")
plt.xlabel("Latency [ms]")
plt.ylabel("Model name")
plt.scatter([x["latency"] for x in test_set], [x["modelName"] for x in test_set], s=100)
plt.locator_params(axis="x", nbins=20)
plt.tight_layout()
plt.savefig(f"plots/test_set_performance_{args.phone.replace(' ', '_')}.png", dpi=500)
plt.show()

# scatter plot results for sampled test set measurements
plt.figure(figsize=(10, 5))
if TITLE:
    plt.title("Sampled Test Set Measurements")
plt.xlabel("Latency [ms]")
plt.ylabel("Model name")
plt.scatter([x["latency"] for x in sampled_test_set], [x["modelName"] for x in sampled_test_set], s=100)
plt.locator_params(axis="x", nbins=20)
plt.tight_layout()
plt.savefig(f"plots/sampled_test_set_performance_{args.phone.replace(' ', '_')}.png", dpi=500)
plt.show()

groups = [1, 10, 100, 1000, 2500, 5000]
# split samples into groups
sample_groups = []
for group in groups:
    sample_groups.append([x for x in samples if x["numberOfSamples"] == group])

# scatter plot results for samples measurements in a single plot
figure, axis = plt.subplots(2, 2, figsize=(16, 8))
if TITLE:
    figure.suptitle("Samples Measurements")
# flatten axis
axis = axis.ravel()
for i in range(4):
    axis[i].set_title(f"N = {groups[i]}")
    axis[i].set_xlabel("Latency [ms]")
    axis[i].set_ylabel("Model name")
    axis[i].scatter([x["latency"] for x in sample_groups[i]], [x["modelName"] for x in sample_groups[i]], s=70)
    axis[i].locator_params(axis="x", nbins=20)

plt.tight_layout()
plt.savefig(f"plots/samples_performance_{args.phone.replace(' ', '_')}.png", dpi=500)
plt.show()
