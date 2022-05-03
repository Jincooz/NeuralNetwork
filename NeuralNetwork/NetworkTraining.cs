using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
namespace NeuralNetwork
{
    public interface NetworkTraining
    {
        public void Train();
        public TrainingData TrainingData { get; set; }
        internal NetworkTraining SetDataAboutNeuralNetwork(ref NeuralNetworkData neuralNetworkData);
    }

    public readonly struct NetworkTrainingStrategy
    {
        public static NetworkTrainingBackpropagation Backpropagation { get; }
    }

    public class NetworkTrainingBackpropagation : NetworkTraining
    {
        private TrainingData _trainingData;
        private NeuralNetworkData _neuralNetworkData;
        TrainingData NetworkTraining.TrainingData { get => _trainingData; set => _trainingData = value; }
        void NetworkTraining.Train()
        {
            for (int i = 0; i < _trainingData.CountDatasets; i++)
            {
                List<Matrix> gradientWeights = new List<Matrix>();
                //same shape as _weightMatrices
                List<Vector> gradientBiases = new List<Vector>();
                //same shape as _biases
                for (int j = 1; j < _neuralNetworkData.AmountOfNodes.Count; j++)
                {
                    gradientWeights.Add(Matrix.Build.Dense(_neuralNetworkData.AmountOfNodes[j], _neuralNetworkData.AmountOfNodes[j - 1], 0));
                    gradientBiases.Add(Vector.Build.Dense(_neuralNetworkData.AmountOfNodes[j], 0));
                }
                List<Vector[]> dataset = _trainingData.DataSets[i];
                for (int j = 0; j < _trainingData.AmountOfRecordsInDataset; j++)
                {
                    var input = dataset[j][0];
                    var desireOutput = dataset[j][1];
                    var networkOutput = _neuralNetworkData.ActivationFunction.ActivationFunction(input);
                    var diference = 2 * (networkOutput - desireOutput);
                    //Console.WriteLine();
                    for (int k = gradientWeights.Count - 1; k >= 0; k--)
                    {
                        //double dCdz = 2 * diference * _activationFunction.ActivationFunctionDerivative((_weightMatrices[k] * _layers[k] + _biases[k])[]);
                        var dCda = Matrix.Build.DiagonalOfDiagonalVector(diference);
                        var dadz = Matrix.Build.DiagonalOfDiagonalVector(_neuralNetworkData.ActivationFunction.ActivationFunctionDerivative(_neuralNetworkData.WeightsMatrices[k] * _neuralNetworkData.Layers[k] + _neuralNetworkData.Biases[k]));
                        var dzdw = Matrix.Build.Dense(0, gradientWeights[k].ColumnCount, 0);
                        for (int h = 0; h < diference.Count; h++)
                        {
                            dzdw = dzdw.InsertRow(h, _neuralNetworkData.Layers[k]);
                        }
                        gradientBiases[k] += (dCda * dadz).Diagonal();
                        gradientWeights[k] += dCda * dadz * dzdw;
                        diference = Vector.Build.Dense(_neuralNetworkData.Layers[k].Count, 0);
                        for (int h = 0; h < _neuralNetworkData.Layers[k].Count; h++)
                        {
                            diference[h] = (dadz * dCda * _neuralNetworkData.WeightsMatrices[k]).Column(h).Sum();
                        }
                    }
                }
                //TODO sum delete
                double sum = 0;
                for (int j = 0; j < gradientWeights.Count; j++)
                {
                    _neuralNetworkData.WeightsMatrices[j] -= gradientWeights[j] / _trainingData.AmountOfRecordsInDataset;
                    _neuralNetworkData.Biases[j] -= gradientBiases[j] / _trainingData.AmountOfRecordsInDataset;
                    sum += _neuralNetworkData.WeightsMatrices[j].RowSums().Sum() + _neuralNetworkData.Biases[j].Sum();
                }
                //Console.WriteLine("Gradient: " + sum.ToString());
            }
        }
        NetworkTraining NetworkTraining.SetDataAboutNeuralNetwork(ref NeuralNetworkData neuralNetworkData)
        {
            _neuralNetworkData = neuralNetworkData;
            return this;
        }
    }
}
