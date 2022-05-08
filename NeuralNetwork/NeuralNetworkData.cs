using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
namespace NeuralNetwork
{
    internal class NeuralNetworkData
    {
        private Hyperparameters _hyperparameters;
        private List<Matrix> _weightMatrices = new List<Matrix>();
        private List<Vector> _biases = new List<Vector>();
        private List<Vector> _layers = new List<Vector>();
        private Vector? _inputLayer;
        private Vector? _outputLayer;
        public List<int> AmountOfNodes { get { return _hyperparameters.AmountOfNodes; } }
        public Vector? InputLayer { get => _inputLayer; set => _inputLayer = value; }
        public Vector? OutputLayer { get => _outputLayer; set => _outputLayer = value; }
        public List<Matrix> WeightsMatrices { get => _weightMatrices; set => _weightMatrices = value; }
        public List<Vector> Biases { get => _biases; set => _biases = value; }
        public List<Vector> Layers { get => _layers; set => _layers = value; }
        public ActivationFunction ActivationFunction { get => _hyperparameters.ActivationFunction; set => _hyperparameters.ActivationFunction = value == null ? ActivationFunctionStrategy.Logistic : value; }
        public Hyperparameters Hyperparameters { get => _hyperparameters; set => _hyperparameters = value; }
        public NetworkTrainingMethod TrainingMethod { get => _hyperparameters.TrainingMethod; set => _hyperparameters.TrainingMethod = value == null ? NetworkTrainingStrategy.Backpropagation : value ; }
        public double LearningRate { get => _hyperparameters.LearningRate ; set => _hyperparameters.LearningRate = value; }
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
            _hyperparameters = new Hyperparameters(new List<int>(amountOfNodes));
            _inputLayer = Vector.Build.Dense(_hyperparameters.AmountOfNodes[0], 0.0);
            _outputLayer = Vector.Build.Dense(_hyperparameters.AmountOfNodes[_hyperparameters.AmountOfNodes.Count - 1], 0.0);
            for (int i = 0; i < _hyperparameters.AmountOfNodes.Count; i++)
            {
                _layers.Add(Vector.Build.Dense(_hyperparameters.AmountOfNodes[i], 0.0));
                if (i != 0)
                {
                    _weightMatrices.Add(XavierWeightInitialization(numberOfNeronsInPreviusLayer: _hyperparameters.AmountOfNodes[i - 1], numberOfNeronsInThisLayer: _hyperparameters.AmountOfNodes[i]));
                    _biases.Add(Vector.Build.Dense(_hyperparameters.AmountOfNodes[i], 0));
                }
            }
        }
    }
}
