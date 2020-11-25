import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # non-negative
        run = self.run(x)
        if (nn.as_scalar(run) >= 0.0):
            return 1
        else: 
        # negative
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            successful = True
            itern = 1
            for a, b in dataset.iterate_once(itern):
                predict = self.get_prediction(a)
                y_value = nn.as_scalar(b)
                if (y_value != predict):
                    successful = False
                    nn.Parameter.update(self.w, a, y_value)
            if successful: 
                break
            

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100
        self.hidden_size = 50
        self.learning_rate = .01

        # weights
        self.w1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 1)
        # bias
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # two layer 
        # f(x) = relu(x*w1 + b1) * w2 + b2
        xw1 = nn.Linear(x, self.w1)
        xw1PlusB1 = nn.AddBias(xw1, self.b1)
        relu = nn.ReLU(xw1PlusB1)
        reluw2 = nn.Linear(relu, self.w2)
        reluw2PlusB2 = nn.AddBias(reluw2, self.b2)

        return reluw2PlusB2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # w * direction < 0, so we use negative multiplier
        multiplier = self.learning_rate * -1
        # params to update
        params = [self.w1, self.w2, self.b1, self.b2]
        loss = float('inf')
        while loss >= .02:
            # retrieve batches of training examples
            for x,y in dataset.iterate_once(self.batch_size):
                # construct a loss node
                loss = self.get_loss(x,y)
                # gradients of the loss with respect to the parameters
                grad_params = nn.gradients(loss, params)
                # get python number for loss
                loss = nn.as_scalar(loss)
                # update our parameters
                for i in range(4):
                    params[i].update(grad_params[i], multiplier)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100
        self.hidden_size = 300
        self.learning_rate = .5

        self.w1 = nn.Parameter(784, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 10)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # two layer 
        # f(x) = relu(x*w1 + b1) * w2 + b2
        xw1 = nn.Linear(x, self.w1)
        xw1PlusB1 = nn.AddBias(xw1, self.b1)
        relu = nn.ReLU(xw1PlusB1)
        reluw2 = nn.Linear(relu, self.w2)
        reluw2PlusB2 = nn.AddBias(reluw2, self.b2)

        return reluw2PlusB2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # w * direction < 0, so we use negative multiplier
        multiplier = self.learning_rate * -1
        # params to update
        params = [self.w1, self.w2, self.b1, self.b2]
        while dataset.get_validation_accuracy() < .97:
            # retrieve batches of training examples
            for x,y in dataset.iterate_once(self.batch_size):
                # construct a loss node
                loss = self.get_loss(x,y)
                # gradients of the loss with respect to the parameters
                grad_params = nn.gradients(loss, params)
                for i in range(4):
                    params[i].update(grad_params[i], multiplier)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # dimension size
        self.dim_size = 5
        self.batch_size = 100
        self.hidden_size = 200
        self.learning_rate = 0.1

        # weights
        self.w = nn.Parameter(self.num_chars, self.hidden_size)
        self.w_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.w_final = nn.Parameter(self.hidden_size, self.dim_size)
        self.b = nn.Parameter(1, self.hidden_size)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # one layer 
        # based on architecture, f(x) = relu(x*w1 + b1)
        # except z_i = x_i*w + h_i * w_hidden
        # so it's more like f(z) = relu(z + b1)
        # z0 = x0 * w
        z_0 = nn.Linear(xs[0], self.w)
        z_0PlusB1 = nn.AddBias(z_0, self.b)
        relu = nn.ReLU(z_0PlusB1)
        # compute first h
        h_i = relu

        for x in xs[1:]:
            xw = nn.Linear(x, self.w)
            hw = nn.Linear(h_i, self.w_hidden)
            # z_i = x_i*w + h_i * w_hidden
            z_i = nn.Add(xw, hw)
            addBias = nn.AddBias(z_i, self.b)
            h_i = nn.ReLU(addBias)
        return nn.Linear(h_i, self.w_final)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # params to update
        params =  [self.w, self.w_hidden, self.w_final]
        multiplier = self.learning_rate * -1
        # stop when accuracy is more than .82 for autograder
        while dataset.get_validation_accuracy() < 0.82:
            # retrieve batches of training examples
            for n, m in dataset.iterate_once(self.batch_size):
                # construct loss node
                getLoss = self.get_loss(n,m)
                # gradients of the loss with respect to the parameters
                gradi = nn.gradients(getLoss, params)
                # update our parameters
                self.w.update(gradi[0], multiplier)
                self.w_hidden.update(gradi[1], multiplier)
                self.w_final.update(gradi[2], multiplier)
