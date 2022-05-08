using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Hyperparameters
    {
        private double _learningRate = 1;
        private readonly List<int> _amountOfNodes;
        private ActivationFunction _activationFunction;
        private NetworkTrainingMethod _trainingMethod;
        public double LearningRate { get => _learningRate; set => _learningRate = value; }
        public List<int> AmountOfNodes { get => _amountOfNodes; }
        public ActivationFunction ActivationFunction { get => _activationFunction; set => _activationFunction = value; }
        public NetworkTrainingMethod TrainingMethod { get => _trainingMethod; set => _trainingMethod = value; }
        internal Hyperparameters(List<int> amountOfNode)
        {
            _amountOfNodes = amountOfNode;
        }
        public Hyperparameters(int[] amountOfNodes, double learningRate = 1, ActivationFunction? activationFunction = null, NetworkTrainingMethod? trainingMethod = null)
        {
            _learningRate = learningRate;
            _activationFunction = activationFunction == null ? ActivationFunctionStrategy.Logistic : activationFunction;
            _trainingMethod = trainingMethod == null? NetworkTrainingStrategy.Backpropagation : trainingMethod;
            _amountOfNodes = new List<int>(amountOfNodes);
        }
    }
}
