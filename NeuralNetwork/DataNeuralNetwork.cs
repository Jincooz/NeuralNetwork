using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using System.Text;
namespace NeuralNetwork
{
    public class DataNeuralNetwork
    {
        private Hyperparameters _hyperparameters;
        private List<Matrix> _weightMatrices = new List<Matrix>();
        private List<Vector> _biases = new List<Vector>();
        internal List<Matrix> WeightMatrices { get { return _weightMatrices; } }
        internal List<Vector> Biases { get { return _biases; } }
        internal Hyperparameters Hyperparameters { get { return _hyperparameters; } }
        internal DataNeuralNetwork(List<Matrix> matrices, List<Vector> vectors, Hyperparameters hyperparameters)
        {
            _weightMatrices = matrices;
            _biases = vectors;
            _hyperparameters = hyperparameters;
        }
        public DataNeuralNetwork()
        {
        }
        public void ToFile(string path)
        {
            using (FileStream stream = File.Create(path))
            {
                string result = "";
                for (int i = 0; i < _hyperparameters.AmountOfNodes.Count; i++)
                {
                    result += _hyperparameters.AmountOfNodes[i].ToString() + " ";
                }
                result = result.Remove(result.Length - 1);
                result += '\n';
                result += _hyperparameters.LearningRate.ToString() + "\n";
                for (int i = 0; i < _weightMatrices.Count; i++)
                {
                    for (int j = 0; j < _weightMatrices[i].RowCount; j++)
                    {
                        for (int k = 0; k < _weightMatrices[i].ColumnCount; k++)
                        {
                            result += _weightMatrices[i][j, k].ToString() + " ";
                        }
                        result = result.Remove(result.Length - 1);
                        result += '\\';
                    }
                    result = result.Remove(result.Length - 1);
                    result += '|';
                }
                result = result.Remove(result.Length - 1);
                result += '\n';
                for (int i = 0; i < _biases.Count; i++)
                {
                    for (int j = 0; j < _biases[i].Count; j++)
                    {
                        result += _biases[i][j].ToString() + " ";
                    }
                    result = result.Remove(result.Length - 1);
                    result += '|';
                }
                byte[] output = new UTF8Encoding(true).GetBytes(result);
                stream.Write(output, 0, output.Length);
            }
        }
        public void FromFile(string path)
        {
            string[] value = File.ReadAllLines(path);
            string[] AmountOfNodes = value[0].Split(' ');
            int[] AmountOfNode = new int[AmountOfNodes.Length];
            for(int i = 0; i < AmountOfNodes.Length; i++)
            {
                AmountOfNode[i] = Convert.ToInt32(AmountOfNodes[i]); 
            }
            double LearningRate = Double.Parse(value[1]);
            _hyperparameters = new Hyperparameters(AmountOfNode, LearningRate, _hyperparameters.ActivationFunction, _hyperparameters.TrainingMethod);
            string[] WeightMatrices = value[2].Split('|');
            _weightMatrices = new List<Matrix>();
            for (int i = 0; i < WeightMatrices.Length; i++)
            {
                string[] WeightMatrix = WeightMatrices[i].Split('\\');
                double[,] weightMatrix = new double[WeightMatrix.Length,WeightMatrix[0].Split(' ').Length];
                for(int j = 0; j < WeightMatrix.Length; j++)
                {
                    string[] Weights = WeightMatrix[j].Split(' ');
                    for(int k = 0; k < Weights.Length; k++)
                    {
                        weightMatrix[j,k] = Convert.ToDouble(Weights[k]);
                    }
                }
                _weightMatrices.Add(Matrix.Build.DenseOfArray(weightMatrix));
            }
            string[] Biases = value[3].Split('|');
            _biases = new List<Vector>();
            for (int i = 0; i < Biases.Length; i++)
            {
                string[] Bias = Biases[i].Split(' ');
                double[] dBias = new double[Bias.Length];
                for (int j = 0; j < Bias.Length; j++)
                {
                    dBias[j] = Convert.ToDouble(Bias[j]); 
                }
                _biases.Add(Vector.Build.DenseOfArray(dBias));
            }
        }
    }
}
