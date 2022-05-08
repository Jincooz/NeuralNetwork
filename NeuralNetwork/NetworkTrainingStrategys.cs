using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
namespace NeuralNetwork
{
    public interface NetworkTrainingMethod
    {
        internal void Train(TrainingData trainingData, ref NeuralNetwork neuralNetwork);
    }

    public readonly struct NetworkTrainingStrategy
    {
        public static NetworkTrainingMethodBackpropagation Backpropagation { get; } = new NetworkTrainingMethodBackpropagation();
    }

    public class NetworkTrainingMethodBackpropagation : NetworkTrainingMethod
    {
        void NetworkTrainingMethod.Train(TrainingData trainingData, ref NeuralNetwork neuralNetwork)
        {
            for (int i = 0; i < trainingData.CountDatasets; i++)
            {
                List<Matrix> gradientWeights = new List<Matrix>();
                //same shape as _weightMatrices
                List<Vector> gradientBiases = new List<Vector>();
                //same shape as _biases
                for (int j = 1; j < neuralNetwork.NeuralNetworkData.AmountOfNodes.Count; j++)
                {
                    gradientWeights.Add(Matrix.Build.Dense(neuralNetwork.NeuralNetworkData.AmountOfNodes[j], neuralNetwork.NeuralNetworkData.AmountOfNodes[j - 1], 0));
                    gradientBiases.Add(Vector.Build.Dense(neuralNetwork.NeuralNetworkData.AmountOfNodes[j], 0));
                }
                List<Vector[]> dataset = trainingData.DataSets[i];
                for (int j = 0; j < trainingData.AmountOfRecordsInDataset; j++)
                {
                    var input = dataset[j][0];
                    var desireOutput = dataset[j][1];
                    var networkOutput = neuralNetwork.ActivateNeuralNetwork(input);
                    var diference = 2 * (networkOutput - desireOutput);
                    //Console.WriteLine();
                    for (int k = gradientWeights.Count - 1; k >= 0; k--)
                    {
                        //double dCdz = 2 * diference * _activationFunction.ActivationFunctionDerivative((_weightMatrices[k] * _layers[k] + _biases[k])[]);
                        var dCda = Matrix.Build.DiagonalOfDiagonalVector(diference);
                        var dadz = Matrix.Build.DiagonalOfDiagonalVector(neuralNetwork.NeuralNetworkData.ActivationFunction.ActivationFunctionDerivative(neuralNetwork.NeuralNetworkData.WeightsMatrices[k] * neuralNetwork.NeuralNetworkData.Layers[k] + neuralNetwork.NeuralNetworkData.Biases[k]));
                        var dzdw = Matrix.Build.Dense(0, gradientWeights[k].ColumnCount, 0);
                        for (int h = 0; h < diference.Count; h++)
                        {
                            dzdw = dzdw.InsertRow(h, neuralNetwork.NeuralNetworkData.Layers[k]);
                        }
                        gradientBiases[k] += (dCda * dadz).Diagonal();
                        gradientWeights[k] += dCda * dadz * dzdw;
                        diference = Vector.Build.Dense(neuralNetwork.NeuralNetworkData.Layers[k].Count, 0);
                        for (int h = 0; h < neuralNetwork.NeuralNetworkData.Layers[k].Count; h++)
                        {
                            diference[h] = (dadz * dCda * neuralNetwork.NeuralNetworkData.WeightsMatrices[k]).Column(h).Sum();
                        }
                    }
                }
                //TODO sum delete
                double sum = 0;
                for (int j = 0; j < gradientWeights.Count; j++)
                {
                    neuralNetwork.NeuralNetworkData.WeightsMatrices[j] -= gradientWeights[j] / trainingData.AmountOfRecordsInDataset;
                    neuralNetwork.NeuralNetworkData.Biases[j] -= gradientBiases[j] / trainingData.AmountOfRecordsInDataset;
                }
                //Console.WriteLine("Gradient: " + sum.ToString());
            }
        }
    }
}
