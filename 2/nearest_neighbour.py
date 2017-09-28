import numpy as np

class1 = [0.4003, 0.3988, 0.3998, 0.3997, 0.4010, 0.3995, 0.3991]
class2 = [0.2554, 0.3139, 0.2627, 0.3802, 0.3287, 0.3160, 0.2924]
class3 = [0.5632, 0.7687, 0.0524, 0.7586, 0.4243, 0.5005, 0.6769]

training = [class1[:4], class2[:4], class3[:4]]

test = [class1[4:], class2[4:], class3[4:]]

nbr_wrong = 0
for row_nbr, row in enumerate(test):
    for item in row:
        items = np.zeros((3,4))
        items.fill(item)
        err = np.abs(training - items)
        classified_as = np.argmin(np.min(err, axis=1))
        if classified_as != row_nbr:
            nbr_wrong += 1
            print("{} is wrongly classified as Class {}".format(item, classified_as + 1))

print("Number of missclassified items are {}".format(nbr_wrong))
