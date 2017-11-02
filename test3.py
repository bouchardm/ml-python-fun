from numpy import exp, array, random, dot, mean, abs, append
import numpy as np


class NeuNet:
    def __init__(self):
        print "Setup 3 neuron input"

        random.seed(1)
        self.synapses = random.random((3, 4))
        self.synapses2 = random.random((4, 1))

        print "Neuron + lien random"
        print self.synapses
        print self.synapses2

    def activation(self, x):
        return 1 / (1 + exp(-x))

    def derivActivation(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs):
        print "tranning 10 000 fois"

        # add bias
        #inputs = array([append(x, [1]) for x in inputs])

        for i in xrange(1000000):
            resultFirstLayer = self.outputWithActivation(inputs, self.synapses) # 5, 4
            resultSecondLayer = self.outputWithActivation(resultFirstLayer, self.synapses2) # 5, 1


            errorSecondLayer = outputs - resultSecondLayer # 5, 1
            ajustementSecondLayer = errorSecondLayer * self.derivActivation(resultSecondLayer) # 5, 1

            errorFirstLayer = dot(ajustementSecondLayer, self.synapses2.T) # 5, 4
            ajustementFirstLayer = errorFirstLayer * self.derivActivation(resultFirstLayer) # 5, 4


            if i % 100000 == 0:
                print mean(abs(errorFirstLayer)), mean(abs(errorSecondLayer))


            # fix error
            self.synapses += inputs.T.dot(ajustementFirstLayer)
            self.synapses2 += resultFirstLayer.T.dot(ajustementSecondLayer)

        print "neuron after"
        print self.synapses
        print self.synapses2


    def outputWithActivation(self, input, synapses):

        return self.activation(dot(input, synapses))


    def think(self, input, asBias = False):

        firstLayer = self.activation(dot(input, self.synapses))

        return self.activation(dot(firstLayer, self.synapses2))


# calls


net = NeuNet()

net.train(array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0]]), array([[0], [1], [1], [0], [0]]))

output = net.think([1, 0, 0])
print "Think!"
print '%.25f' % output


output = net.think([0, 0, 0])
print "Think!"
print '%.25f' % output




# bias test
