# NaiveBayesClassifier

A cleaned-up, extended version of a Naive Bayes Classifier I wrote for my Machine Learning class at UTD.

## Intro to Bayesian Classification

As the name suggests, this particular Machine Learning algorithm is used for classification tasks. That is, it is an algorithmic, dynamic solution for attempting to categorize data into certain labels.

Bayesian Learning is a *supervised algorithm*, meaning it only operates if first fed training data by a "teacher". Training data is of the same form as the desired data to be classified (that is, if you want to classify songs, your training data should consist of songs), and is already labeled by the teacher. These data points consist of attributes, each of which takes on a certain value.

A supervised algorithm mathematically analyzes all of this training data and attempts to construct a model based on the data provided which provides a simulation of reality. That is to say, the algorithm attempts to understand the fundamental "rules" of the classification task by peeking at some of the answers.

We then apply this data to unclassified testing data, which has the same attributes as the training data, but no pre-written classification. The Classifier then applies this model to the unlabeled data and attempts to determine which class label is most probable.

Bayesian classification is rooted in Bayes' Rule, a probability law that goes as follows: 

P(A|B) = P(B|A)P(A)/P(B).

In English this means that the probability of event A occuring, given that event B has occured, is equal to the reverse, times the probability of event A occuring, divided by the probability of event B occuring. Generally speaking, Bayes' Rule is a good way to reconstruct conditional probabilities into consituent probabilities.

In a classification problem, you have a key "event", C, which is the class label, you're trying to predict probabilistically, and a bunch of events, which are the attributes, which we'll call X, Y, and Z, but could really be any number of attributes. These attributes are given for every instance, so what we're really trying to predict is the probability of any given class label. That is, P(C|X, Y, Z)

## The Naive Assumption

The key aspect of the name of this method we haven't touched on yet is "naive". You see, when we apply Bayes' Rule to P(C|X, Y, Z), we get this statement:

P(C|X, Y, Z) = P(X, Y, Z|C)P(C)/P(X, Y, Z), for every class C.

We have to complete this formula for every class label and compare them, and we'll select the highest. Since all of these statements will have the same denominator, it won't affect which one is the highest, so we drop it, geting this:

P(X, Y, Z|C)P(C)

Here's where we make a *naive* assumption: We assume that all of the attributes of the testing instance are independent (that is, the presence of one doesn't affect the probability of another), given the class. For a variety of reasons, this is a pretty silly thing to assume, but it *tends* to produce mathematically valid answers in the end, and it makes the math simpler. You see, if you have P(A, B|C), and A and B are conditionally independent given C, that statement equals P(A|C)P(B|C). So, we apply this logic to the above statement and get:

P(X|C)P(Y|C)P(Z|C)P(C)

This is a formula composed of very easily obtained information. P(C) is just the number of training instances of class C divided by the total number of training instances (remember, we're using what we observe to form our understanding of the world!) and P(Attribute|C) is found by dividing the total number of class C training instances with attribute value Attribute, by the number of training instances of class C.

### A Technicality

This statement produces a likelihood, technically. It's still a measurement of the mathematical chances of a given class label being true, but all possible class labels' probabilities should sum to 1, to satisfy the basic laws of probability (after all, a training instance has to have *some* class label). However, were we to calculate this quantity for every class label and sum them, we'd get some random number. To get probabilities, we could just divide each likelihood by the sum of all likelihoods. The final answer is the same.

## The Code In Action

So, this implementation of this algorithm relies heavily on an OO framework. We have a single massive object called the Problem Space, which contains information encompassing all of the instances, including the total number of training instances, the total number of instances of each class, and the names of the classes. 

Contained within the Problem Space is a list of Attribute objects, each representing a single attribute measured in each instance. Attributes have names, and they themselves contain a list of AttValue objects, each representing a value that said Attribute can take on. These AttValues have names and a list of what are called "class constitutions". Essentially, these are the number of instances of a given class which have this particular value for this particular attribute. 

For example, if we have 100 instances of the class "person", and we have an attribute called "eye color" with an AttValue called "blue", the class_constitution[person] of that AttValue is simply the number of observed people with blue eyes.

The algorithm begins by going over the training instances and using the data to construct the Problem Space, Attributes, and AttValues. These three things in concert form our model. It reads the first line of the training file, which is expected to be a set of attribute headers, and uses those to build every Attribute object it will need. As the program processes training instances, if it comes across an AttValue or a class it doesn't recognize, it will dynamically add it to the Problem Space.

When it comes time to classify, all of the various numbers needed to calculate likelihoods and probabilities are contained within the Problem Space, and are easily called upon. We just look at every attribute value of the testing instance, use our training data to construct the P(X|C)P(Y|C)P(Z|C)P(C)-like equation needed to calculate the likelihood of a given class, do that for every class, then assign the instance the class with the highest likelihood.

## To-Do List:

* Allow the classifier to operate on .txt, .csv, and .json files. File type can be determined with regex, at which point the file name will be passed to distinct functions for processing. Depending on implementation, a class-based/interface-based set of services might save me some time.
* Allow the classifier to be less dependent on the exact values of any given attribute by using a k-nearest neighbors powered comparitor. If we don't have the exact value of the attribute in the problem space, we use a similarity function to find the attribute values closest the the actual, and using that data to classify.
