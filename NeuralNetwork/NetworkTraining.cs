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
        private NetworkTrainingMethod _trainingMethod;
        public NetworkTraining(ref NeuralNetwork neuralNetwork, TrainingData trainingData, NetworkTrainingMethod? networkTrainingMethod = null)
        {
            _trainingData = trainingData;
            _neuralNetwork = neuralNetwork;
            if (networkTrainingMethod != null)
                _trainingMethod = networkTrainingMethod;
            else
                _trainingMethod = NetworkTrainingStrategy.Backpropagation;
        }
        public void SetTrainingStrategy(NetworkTrainingMethod trainingMethod)
        {
            _trainingMethod = trainingMethod;
        }
        public void Train()
        {
            _trainingMethod.Train(_trainingData,ref _neuralNetwork);
        }
    }
}
