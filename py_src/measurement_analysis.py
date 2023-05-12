import json
import matplotlib.pyplot as plt

# object properties: latency (float), numberOfSamples (int), predictionType (string: SAMPLES, TEST_SET, SAMPLED_TEST_SET), 
#                    modelName (string)

with open("measurements.json", "r") as file:
    data = json.load(file)

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

# scatter plot results for test set measurements
plt.figure(figsize=(14, 6))
plt.title("Test set measurements")
plt.xlabel("Latency (ms)")
plt.ylabel("Model name")
plt.scatter([x["latency"] for x in test_set], [x["modelName"] for x in test_set], s=100)
plt.locator_params(axis="x", nbins=24)
plt.tight_layout()
plt.savefig("plots/test_set_performance.png", dpi=500)
plt.show()

# scatter plot results for sampled test set measurements
plt.figure(figsize=(14, 8))
plt.title("Sampled test set measurements")
plt.xlabel("Latency (ms)")
plt.ylabel("Model name")
plt.scatter([x["latency"] for x in sampled_test_set], [x["modelName"] for x in sampled_test_set], s=100)
plt.locator_params(axis="x", nbins=24)
plt.tight_layout()
plt.savefig("plots/sampled_test_set_performance.png", dpi=500)
plt.show()

groups = [1, 10, 100, 1000, 2500, 5000, 7500]
# split samples into groups
sample_groups = []
for group in groups:
    sample_groups.append([x for x in samples if x["numberOfSamples"] == group])

# scatter plot results for samples measurements
for i in range(len(sample_groups)):
    plt.figure(figsize=(14, 8))
    plt.title("Samples measurements (n = {})".format(groups[i]))
    plt.xlabel("Latency (ms)")
    plt.ylabel("Model name")
    plt.scatter([x["latency"] for x in sample_groups[i]], [x["modelName"] for x in sample_groups[i]], s=100)
    plt.locator_params(axis="x", nbins=24)
    plt.tight_layout()
    plt.savefig("plots/samples_performance_{}.png".format(groups[i]), dpi=500)
    plt.show()