from numpy import exp, array, random, dot, mean, abs, append


class NeuNet:
    def __init__(self):
        print "Setup 3 neuron input"

        random.seed(1)
        self.neurons = random.random((4, 1))

        print "Neuron + lien random"
        print self.neurons

    def activation(self, x):
        return 1 / (1 + exp(-x))

    def derivActivation(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs):
        print "tranning 10 000 fois"

        # add bias
        inputs = array([append(x, [1]) for x in inputs])

        for i in xrange(1000000):
            result = self.think(inputs, True)

            error = outputs - result

            if i % 10000 == 0:
                print mean(abs(error))

            # fix error
            ajustement = dot(inputs.T, error * self.derivActivation(result))

            self.neurons += ajustement

        print "neuron after"
        print self.neurons



    def think(self, input, asBias = False):
        if asBias == False:
            input = append(input, [1])
        return self.activation(dot(input, self.neurons))


# calls


net = NeuNet()

net.train(array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]), array([[0], [1], [1], [0]]))

output = net.think([1, 0, 0])
print "Think!"
print output


output = net.think([0, 0, 0])
print "Think!"
print output




# bias test
