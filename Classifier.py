__author__ = 'Michael Burdick'

import sys


# AttValue is an object representing a single value that an Attribute may take
class AttValue:
    def __init__(self, value_name = ""):
        self.label = value_name  # contains the name of the attribute value, NOT the name of the attribute
        self.class_constitution = []  # the number of training instances of each class which have this particular
        # attribute set to this particular attribute value. So, if there are 10 training instances of class 0,
        # and 3 of them have their first attribute set to 1, then for the attValue "1" for the first attribute,
        # class_constitution[0] = 3

    def get_name(self):
        return self.label

    def increment_class_constitution(self, class_index):
        # This function increments the appropriate class counter. Keep in mind, we're keeping track of how many
        # training instances of each class take this value for this attribute
        try:
            self.class_constitution[class_index] += 1
        except IndexError:
            # Sometimes, we just won't have updated our list of classes to include a class we just found out about.
            # If that's the case, then we update the list first, then update
            for y in range(len(self.class_constitution) - 1, class_index):
                self.class_constitution.append(0)
            self.class_constitution[class_index] += 1

    def get_class_constitution(self):
        return self.class_constitution

    def print(self):
        print(self.label)
        print(self.class_constitution)

# Attribute represents a single attribute of a data point in the set
# Attributes consist of a list of AttValues
class Attribute:
    def __init__(self, attribute_name):
        # We initialize our attributes with just their names, we'll add in values later in the learning process
        self.name = attribute_name
        self.list_of_values = []

    def update_values(self, attribute_value, class_index):
        # update_values takes in a value for this attribute, and the class of the current training instance, and
        # updates the class_constitution of the appropriate attValue object
        for value in self.list_of_values:
            # We start by simply checking to see if the appropriate AttValue exists, and if so, we update it
            if value.get_name() == attribute_value:
                value.increment_class_constitution(class_index)
                return
        # If we get through all of the AttValues and can't find the one we want, we make it, then update it
        self.list_of_values.append(AttValue(attribute_value))
        self.list_of_values[-1].increment_class_constitution(class_index)

    def get_value_counts(self, attribute_value):
        for value in self.list_of_values:
            if value.get_name() == attribute_value:
                return value.get_class_constitution()
        return [1]

    def print(self):
        print(self.name)
        for value in self.list_of_values:
            value.print()


# ProblemSpace is a representation of the entire classification problem
# It contains every attribute, which in turn contains every attribute value, and also has the pure class probabilities
class ProblemSpace:
    def __init__(self, list_of_attributes):
        # when we initialize, we take the first row of the input data (the attribute names), and use them to construct
        # a list of every attribute. These will remain empty at first, as will our key of class names, and our list
        # of class probabilities.
        self.total_training_instances = 0
        self.class_names = []
        self.class_counts = []
        self.attributes = []
        for attribute_name in list_of_attributes:
            self.attributes.append(Attribute(attribute_name))

    def get_class_index(self, class_name):
        # sometimes we'll have the class name, and need to know which index of a probability list to update
        # this function returns the index associated with that class. If no such index exists, it just returns -1
        try:
            index = self.class_names.index(class_name)
        except ValueError:
            self.class_names.append(class_name)
            self.class_counts.append(0)
            index = len(self.class_names)-1
        return index

    def increment_class(self, class_name):
        # when we process a new training instance, one of the first things we'll wanna do is to update the total class
        # counts, and in fact the total training instance count, to reflect the new addition. This function does that
        self.total_training_instances += 1
        class_index = self.get_class_index(class_name)
        self.class_counts[class_index] += 1
        return class_index

    def update_attribute(self, attribute_number, attribute_value, class_index):
        # update_attribute is called to update the current attribute with the new value count
        # this is handled within the attribute, not the problem space, so we just pass the info along as needed
        self.attributes[attribute_number].update_values(attribute_value, class_index)

    def generate_class_probabilities(self):
        # This function uses total_training_instances and the class_counts array to generate the P(C) for every class
        class_probabilities = []
        denominator = 0

        # We start out by creating an array of values of the form (# of values of class X/total number of instances)
        # Note that these are likelihoods, not probabilities; they don't sum to 1
        # We are going to keep track of the sum of these likelihoods, though
        for class_count in self.class_counts:
            likelihood = class_count/self.total_training_instances
            class_probabilities.append(likelihood)
            denominator += likelihood

        # Once we have an array of likelihoods, we turn them into probabilities by dividing each by the sum of array
        for value in class_probabilities:
            value = value/denominator

        return class_probabilities

    def get_class_totals(self):
        return self.class_counts

    def retrieve_conditional_probabilities(self, attribute_number, attribute_value):
        return self.attributes[attribute_number].get_value_counts(attribute_value)

    def get_class_names(self):
        return self.class_names

    def print_problem_space(self):
        print("Total Number of Training Instances: " + str(self.total_training_instances))
        for i in range(len(class_names)):
            print("Class " + str(i) + ": " + self.class_names[i] + " - " + str(self.class_counts[i]) + "/" + str(self.total_training_instances))
        for attribute in self.attributes:
            attribute.print()


# ------------------------- Start of Main ------------------------- #

# The program runs with two arguments, the name of the training file, and the name of the testing file
# If we start with an improper number of arguments, the whole thing just ends.
if len(sys.argv) != 3:
    print("Found %s arguments, expecting 2." %(len(sys.argv) - 1))
    print("Exiting")
    sys.exit()

# First, we go ahead and pull in those filenames and save them as variables
training_file_name = sys.argv[1]
testing_file_name = sys.argv[2]

# This marks the start of the TRAINING process, which begins, undramatically, by opening the training file
training_file = open(training_file_name, "r")

# Next, we extract the first line of the training file, which should be attribute names
# We go ahead and split them into a list, splitting on the default argument (which is a space)
attribute_names = training_file.readline().split()[:-1]

# Then, we construct a ProblemSpace object, which in turn constructs Attribute objects, based on this list
problem = ProblemSpace(attribute_names)

# When we have that, we start extracting training instances and using them to train
for line in training_file:
    # these first three lines simply clean up the input and prepare it for processing
    instance_attributes = line.split()
    instance_class = instance_attributes[-1]
    instance_attributes = instance_attributes[:-1]

    # now, we're gonna increment the appropriate class counter, as well as the total count of training instances
    class_index = problem.increment_class(instance_class)

    for x in range(len(instance_attributes)):
        # This for-loop exists to update AttValues with appropriate counts for this instance's attribute values
        # We're using a C-style iterator here so that we can direct ProblemSpace to the precise Attribute it needs
        # to be updating, rather than forcing it to do a search.

        problem.update_attribute(x, instance_attributes[x], class_index)

training_file.close()

# By this point, we've run entirely through the training instances. Radical!
# Now, it's time to begin CLASSIFICATION

testing_file = open(testing_file_name, "r")
output_file = open("classifications.txt", "w")

# We need to pull the pure probabilities of for each class, as well as the total number of training instances of
# each class
total_class_probabilities = problem.generate_class_probabilities()
class_totals = problem.get_class_totals()
class_names = problem.get_class_names()

# problem.print_problem_space()

for line in testing_file:
    # First, we prepare the attributes of the testing instance for processing
    instance_attributes = line.split()

    #DEBUG: statement_builder is done to see the numbers the classifier is crunching
    statement_builder = [""]*len(class_names)

    # Then, we create likelihoods_of_class to store the likelihoods that this testing instance belongs to any one class
    # We're using Naive Bayes, so the product which becomes said likelihoods includes the total P(C). So, it's sensible
    # to start off with those P(C) values already in there
    likelihoods_of_class = total_class_probabilities[:]

    for x in range(len(instance_attributes)):
        attribute_value_counts = problem.retrieve_conditional_probabilities(x, instance_attributes[x])
        for z in range(len(class_names)):
            try:
                likelihoods_of_class[z] *= attribute_value_counts[z]/class_totals[z]
            except IndexError:
                likelihoods_of_class[z] *= 0

    maximum_likelihood = 0
    index_of_max = -1
    for q in range(len(likelihoods_of_class)):
        if likelihoods_of_class[q] >= maximum_likelihood:
            maximum_likelihood = likelihoods_of_class[q]
            index_of_max = q

    output_file.write(class_names[index_of_max] + "\n")

output_file.close()