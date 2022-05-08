using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NetworkTraining
    {
        private TrainingData _trainingData;
        private NeuralNetwork _neuralNetwork;
        public NetworkTraining(ref NeuralNetwork neuralNetwork, TrainingData trainingData, NetworkTrainingMethod? networkTrainingMethod = null)
        {
            _trainingData = trainingData;
            _neuralNetwork = neuralNetwork;
            _neuralNetwork.NeuralNetworkData.TrainingMethod = networkTrainingMethod;
        }
        public void SetTrainingStrategy(NetworkTrainingMethod trainingMethod) =>
            _neuralNetwork.NeuralNetworkData.TrainingMethod = trainingMethod;
        public void SetLearningRate(double learningRate) =>
            _neuralNetwork.NeuralNetworkData.LearningRate = learningRate;
        public void Train()
        {
            _neuralNetwork.NeuralNetworkData.TrainingMethod.Train(_trainingData,ref _neuralNetwork);
        }
    }
}
