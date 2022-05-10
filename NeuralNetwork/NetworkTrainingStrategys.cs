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
            double past_result = 1;
            int count = 0;
            List<Matrix> gradientWeights = new List<Matrix>();
            //same shape as _weightMatrices
            List<Vector> gradientBiases = new List<Vector>();
            //same shape as _biases
            for (int j = 1; j < neuralNetwork.NeuralNetworkData.AmountOfNodes.Count; j++)
            {
                gradientWeights.Add(Matrix.Build.Dense(neuralNetwork.NeuralNetworkData.AmountOfNodes[j], neuralNetwork.NeuralNetworkData.AmountOfNodes[j - 1], 0));
                gradientBiases.Add(Vector.Build.Dense(neuralNetwork.NeuralNetworkData.AmountOfNodes[j], 0));
            }
            while (true)
            {
                foreach (Matrix matrix in gradientWeights)
                    matrix.Multiply(0);
                foreach (Vector vector in gradientBiases)
                    vector.Multiply(0);
                List<Vector[]> dataset = trainingData.Data;
                for (int i = 0; i < trainingData.AmountOfData; i++)
                {
                    var input = dataset[i][0];
                    var desireOutput = dataset[i][1];
                    var networkOutput = neuralNetwork.ActivateNeuralNetwork(input);
                    var diference = 2 * (networkOutput - desireOutput);
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
                for (int i = 0; i < gradientWeights.Count; i++)
                {
                    gradientWeights[i] = neuralNetwork.NeuralNetworkData.LearningRate * gradientWeights[i];
                    gradientBiases[i] = neuralNetwork.NeuralNetworkData.LearningRate * gradientBiases[i];
                    neuralNetwork.NeuralNetworkData.WeightsMatrices[i] -= gradientWeights[i] / trainingData.AmountOfData;
                    neuralNetwork.NeuralNetworkData.Biases[i] -= gradientBiases[i] / trainingData.AmountOfData;
                    
                }
                double result = 0;
                for (int i = 0; i < trainingData.AmountOfData; i++)
                {
                    var input = dataset[i][0];
                    var desireOutput = dataset[i][1];
                    var networkOutput = neuralNetwork.ActivateNeuralNetwork(input);
                    var diference = (networkOutput - desireOutput);
                    diference = diference.PointwisePower(2);
                    result += diference.Sum();
                }
                result /= trainingData.AmountOfData;
                //if (Math.Abs(result - past_result) < 0.00001)
                //if(result < 0.15)
                //    {
                //        System.Diagnostics.Debug.WriteLine($"Abs now: {past_result} count: {count}");
                //        return;
                //    }
                past_result = result;
                if(++count > 5000)
                {
                    System.Diagnostics.Debug.WriteLine($"!!!Abs now: {past_result} count: {count}");
                    return;
                }

            }
        }
    }
}
