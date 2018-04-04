# coding=UTF-8
from __future__ import division
import re

# This is a naive text summarization algorithm
# April, 2013


class SummaryTool(object):

    # Naive method for splitting a text into sentences
    def split_content_to_sentences(self, content):
        content = content.replace("\n", ". ")
        return content.split(". ")

    # Naive method for splitting a text into paragraphs
    def split_content_to_paragraphs(self, content):
        return content.split("\n\n")

    # Caculate the intersection between 2 sentences
    def sentences_intersection(self, sent1, sent2):

        # split the sentence into words/tokens
        s1 = set(sent1.split(" "))
        s2 = set(sent2.split(" "))

        # If there is not intersection, just return 0
        if (len(s1) + len(s2)) == 0:
            return 0

        # We normalize the result by the average number of words
        return len(s1.intersection(s2)) / ((len(s1) + len(s2)) / 2)

    # Format a sentence - remove all non-alphbetic chars from the sentence
    # We'll use the formatted sentence as a key in our sentences dictionary
    def format_sentence(self, sentence):
        sentence = re.sub(r'\W+', '', sentence)
        return sentence

    # Convert the content into a dictionary <K, V>
    # k = The formatted sentence
    # V = The rank of the sentence
    def get_senteces_ranks(self, content):

        # Split the content into sentences
        sentences = self.split_content_to_sentences(content)

        # Calculate the intersection of every two sentences
        n = len(sentences)
        values = [[0 for x in range(n)] for x in range(n)]
        for i in range(0, n):
            for j in range(0, n):
                values[i][j] = self.sentences_intersection(sentences[i], sentences[j])

        # Build the sentences dictionary
        # The score of a sentences is the sum of all its intersection
        sentences_dic = {}
        for i in range(0, n):
            score = 0
            for j in range(0, n):
                if i == j:
                    continue
                score += values[i][j]
            sentences_dic[self.format_sentence(sentences[i])] = score
        return sentences_dic

    # Return the best sentence in a paragraph
    def get_best_sentence(self, paragraph, sentences_dic):

        # Split the paragraph into sentences
        sentences = self.split_content_to_sentences(paragraph)

        # Ignore short paragraphs
        if len(sentences) < 2:
            return ""

        # Get the best sentence according to the sentences dictionary
        best_sentence = ""
        max_value = 0
        for s in sentences:
            strip_s = self.format_sentence(s)
            if strip_s:
                if sentences_dic[strip_s] > max_value:
                    max_value = sentences_dic[strip_s]
                    best_sentence = s

        return best_sentence

    # Build the summary
    def get_summary(self, title, content, sentences_dic):

        # Split the content into paragraphs
        paragraphs = self.split_content_to_paragraphs(content)

        # Add the title
        summary = []
        summary.append(title.strip())
        summary.append("")

        # Add the best sentence from each paragraph
        for p in paragraphs:
            sentence = self.get_best_sentence(p, sentences_dic).strip()
            if sentence:
                summary.append(sentence)

        return ("\n").join(summary)


# Main method, just run "python summary_tool.py"
def main():
    title = """
    What is a neural network? Can you tell me in an easy language?
    """

    content = """
    Firstly, there are many structures of Neural Network performing different functions and trained in different ways. I'll assume that you're talking about Feed-Forward Backpropagation Neural Networks, as they're most commonly meant when not being more specific.

Firstly, you have Nodes. These take input values, add them together and apply an Activation Function to create an output value. Then you have Connections with weights - the value being passed along that Connection is multiplied by the weight. A weight of 1 will therefore pass on the starting value unchanged, a weight of 0 will always pass on 0, and a weight of -1 will pass on the negative version of the value. A Neural network is nothign more than a whole set of nodes connected to one another. These are logically in layers, where all the Nodes in a given layer have input Connections from the preceding layer and output Connections to the following layer. The first layer (Input) receives the input values directly from the data. The final layer (Output) outputs are the values you went - if you're looking to predict a single classification or a single value (such as the expected next day's Exchange Rate) then the Output layer will have a single Node.

In the original kind of neural network - Perceptrons - the Activation Function was a simple threshold - if the total from all the input connections was above a given threshold it output 1 and otherwise it returned 0. This was modelled on the idea of the brain's synapses reaching a firing threshold, but proves to be difficult to train because it's a sudden jump in response and can't easily be gradually shifted to the correct output. Most networks therefore use one of a range of more sophisticated functions to create a more nuanced response. The simplest mapping, ReLU, just replaces any negative values with 0. Others, like Sigmoid and TanH look like an S-shape when you chart input values to outputs as they map values from negative infinity to positive infinity to the range 0 to 1 or -1 to 1. These functions help ensure that one single value won't drown out all others in subsequent nodes.

As others have noted, Linear and Logistic regressions can be represented as a Neural Network with an Input layer and an Output layer and no layers in between, as the weights of a such a regression become the weights of the connections. However, the value of a Neural Network comes when you have one or more layers in between (known as Hidden Layers because you neither see the inputs they're fed nor their outputs directly). These hidden layers make a Neural Network a universal approximator - meaning that given sufficient nodes it can represent any required shape of function output, at least approximately.

The model is initialised with random connection weights (they need to have different values from one another so each node learns different things about the data during training). The Feed-Forward part takes the input values and feeds them forward through the network, applying the Connection weights and Node activation functions to produce the inputs for the next layer in the network until the Output values are obtained. The Back-Propagation element takes those outputs value and compares it to the correct output values and determines the scale and direction of the errors. Each layer and its onward connections amends the weights to shift the calculated values for that layer towards the true ones. That layers new tweaked outputs are then fed back to the previous layer so it can amend its weights, until the Input layer is reached. This is one training epoch. The system can then run through the data again and again to keep refining its weights to match the inputs.

There are various ways this can be modified - how much do you modify weights each time? Do you give it all of the data each time or random subsets? How many epochs do you run? What are you using to measure the level of error in the first place? This is where hyper-parameters come in. An example would be a network taking 5 input variables (for example, the previous 5 days closing prices for an Exchange Rate, giving you a 5-Node Input Layer), one Hidden Layer with 10 Nodes and a single Node output Layer (e.g. to predict the next-day's Exchange Rate). This would be described as a 5-10-1 layer. However, you could have different activation functions on the layers, run 5, 10 or 100 training epochs on it, use all your data for each training run or just random samples from it, so there's still many things that will impact its performance. When it starts (with just random performance) the output will be nonsense.

One key detail is about over-fitting. However good your data is, unless it is perfectly representative and noise-free (so the only things affecting the output value are changes present in the input values, nothing else), then your target won't be to be 100% correct at predicting your training data, because that would be fitting the noise in the input values. The result is a worse performance when you give it previously unseen data. You have to train and learn enough to pick up as much of the valuable information about interpreting the inputs as possible, while trying to minimise the amount it tries to predict noise. More data may help this, as over a larger dataset the real information should become more apparent, but limiting the number of training epochs may help along with not having too many Nodes (to force a simpler function shape to be calculated) and using random sampling so for any given training run the noise element should vary more than the real information. Sometimes the best approach is to take the model after each training epoch and test it against a separate (held-out) Validation dataset. If the results are better than the previous epoch, keep training, otherwise stop.

The above are just the very basics, deliberately leaving out implementation details like the maths and range of training functions. Other kinds of neural networks have very different structures. Still others don’t use backpropagation - some use Genetic Algorithms to try lots of different weights in separate runs and compare and combine each variant to find improvements - another subject on its own. The above structure is good for learning a function mapping, but others are better for doing things like recognition tasks. Optimisation may require multiple training runs (because learning can get stuck in areas where the changing the value in any direction doesn't improve the output accuracy - a local optimum - but there's another area elsewhere that still scores better - the global optimum). Different structures need to be tried, and hyper-parameters. Input values may need some pre-processing for best results. Training time may be a factor. Some data problems are vast - with Petabytes or more of data, requiring fast processing and learning without being able to keep using the whole dataset. Computer Vision and Speech Recognition and Customer Churn Prediction or Customers Likes identification and other tasks can all be handled by Neural Networks (and their more complex cousins - Deep Belief Networks), but they all need different structures.

If you take a Master's on the subject, you'll work with a number of such models but you'll still only have scratched the surface for how to use them. Hopefully the above explanation gives a taste - every time I learn more about them the more I realise I don't know, and I think most of us are still at that stage - with ever-increasing amounts that we know we don't know rather than feeling we're getting to know everything about Neural Networks. It's a big topic - it can't be covered in full depth even in a book - several books at least.





    """

    # Create a SummaryTool object
    st = SummaryTool()

    # Build the sentences dictionary
    sentences_dic = st.get_senteces_ranks(content)

    # Build the summary with the sentences dictionary
    summary = st.get_summary(title, content, sentences_dic)

    # Print the summary
    print (summary)

    # Print the ratio between the summary length and the original length
    print ("")
    print ("Original Length %s" % (len(title) + len(content)))
    print ("Summary Length %s" % len(summary))
    print ("Summary Ratio: %s" % (100 - (100 * (len(summary) / (len(title) + len(content))))))


if __name__ == '__main__':
    main()
