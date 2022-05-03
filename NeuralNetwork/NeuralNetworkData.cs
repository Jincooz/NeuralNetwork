using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
namespace NeuralNetwork
{
    internal class NeuralNetworkData
    {
        private ActivationFunction _activationFunction;
        private readonly List<int> _amountOfNodes;
        private List<Matrix> _weightMatrices = new List<Matrix>();
        private List<Vector> _biases = new List<Vector>();
        private List<Vector> _layers = new List<Vector>();
        private Vector? _inputLayer;
        private Vector? _outputLayer;
        public List<int> AmountOfNodes { get { return _amountOfNodes; } }
        public Vector? InputLayer { get => _inputLayer; set => _inputLayer = value; }
        public Vector? OutputLayer { get => _outputLayer; set => _outputLayer = value; }
        public List<Matrix> WeightsMatrices { get => _weightMatrices; set => _weightMatrices = value; }
        public List<Vector> Biases { get => _biases; set => _biases = value; }
        public List<Vector> Layers { get => _layers; set => _layers = value; }
        public ActivationFunction ActivationFunction { get => _activationFunction; set => _activationFunction = value; }
        private Matrix XavierWeightInitialization(int numberOfNeronsInPreviusLayer, int numberOfNeronsInThisLayer)
        {
            //uniform probability distribution in (-1/sqrt(n),1/sqrt(n)), where n - amount of neuron in previuse layer
            Random rand = new Random(DateTime.Now.Second);
            double lower = -(1.0 / Math.Sqrt(numberOfNeronsInPreviusLayer)), upper = (1.0 / Math.Sqrt(numberOfNeronsInPreviusLayer));
            var matrix = Matrix.Build.Dense(numberOfNeronsInThisLayer, numberOfNeronsInPreviusLayer, 0);
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] = lower + rand.NextDouble() * (upper - lower);
                }
            }
            return matrix;
        }
        public NeuralNetworkData(int[] amountOfNodes)
        {   
            _amountOfNodes = new List<int>(amountOfNodes);
            _inputLayer = Vector.Build.Dense(_amountOfNodes[0], 0.0);
            _outputLayer = Vector.Build.Dense(_amountOfNodes[_amountOfNodes.Count - 1], 0.0);
            for (int i = 0; i < _amountOfNodes.Count; i++)
            {
                _layers.Add(Vector.Build.Dense(_amountOfNodes[i], 0.0));
                if (i != 0)
                {
                    _weightMatrices.Add(XavierWeightInitialization(numberOfNeronsInPreviusLayer: _amountOfNodes[i - 1], numberOfNeronsInThisLayer: _amountOfNodes[i]));
                    _biases.Add(Vector.Build.Dense(_amountOfNodes[i], 0));
                }
            }
        }
    }
}
