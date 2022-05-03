using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private NetworkTraining _networkTraining;
        private NeuralNetworkData _networkData;
        public Vector? InputLayer { get => _networkData.InputLayer; set => _networkData.InputLayer = value; }
        public Vector? OutputLayer { get => _networkData.OutputLayer; }
        public List<Matrix> WeightsMatrices { set => _networkData.WeightsMatrices = value; }
        internal NeuralNetworkData NeuralNetworkData => _networkData;
        public NeuralNetwork(int[] amountOfNodes, ActivationFunction? activationFunctionStrategy = null, NetworkTraining? networkTrainingStrategy = null)
        {
            _networkData = new NeuralNetworkData(amountOfNodes);
            _networkData.ActivationFunction = ActivationFunctionStrategy.Logistic;
            if (activationFunctionStrategy != null)
                _networkData.ActivationFunction = activationFunctionStrategy;
            if (networkTrainingStrategy != null)
                _networkTraining = networkTrainingStrategy;
        }
        public NeuralNetwork SetActivationStrategy(ActivationFunction activationFunctionStrategy)
        {
            //chose strategy ActivationFunctionStrategys
            _networkData.ActivationFunction = activationFunctionStrategy;
            return this;
        }
        public Vector? ActivateNeuralNetwork(double[] inputLayer)
        {
            _networkData.InputLayer = Vector.Build.DenseOfArray(inputLayer);
            return ActivateNeuralNetwork();
        }
        internal Vector? ActivateNeuralNetwork(Vector? input)
        {
            _networkData.InputLayer = input;
            return ActivateNeuralNetwork();
        }
        public Vector? ActivateNeuralNetwork()
        {
            if (_networkData.InputLayer == null)
                return null;
            _networkData.Layers[0] = _networkData.InputLayer;
            for (int i = 0; i < _networkData.Layers.Count - 1; i++)
            {
                Vector value = _networkData.WeightsMatrices[i] * _networkData.Layers[i] + _networkData.Biases[i];
                _networkData.Layers[i + 1] = _networkData.ActivationFunction.ActivationFunction(value);
            }
            _networkData.OutputLayer = _networkData.Layers[_networkData.Layers.Count - 1];
            return _networkData.OutputLayer;
        }
    }
}
