using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
namespace NeuralNetwork
{
    public class TrainingData
    {
        private readonly int _inputNodesAmount, _outputNodesAmount;
        private readonly List<Vector[]> _data;
        private int _amountOfData;
        public int AmountOfData { get => _amountOfData; }
        public List<Vector[]> Data { get => _data; }
        public TrainingData(Matrix trainingData, int inputNodesAmount, int outputNodesAmount)
        {
            _inputNodesAmount = inputNodesAmount;
            _outputNodesAmount = outputNodesAmount;
            _amountOfData = 0;
            _data = new List<Vector[]>();
            AddData(trainingData);
        }
        public void AddData(Matrix newTrainingData)
        {
            for (int i = 0; i < newTrainingData.RowCount; i++)
            {
                Vector[] newRecord = new Vector[2];
                newRecord[0] = Vector.Build.Dense(_inputNodesAmount, 0);
                newRecord[1] = Vector.Build.Dense(_outputNodesAmount, 0);
                for (int j = 0; j < _inputNodesAmount + _outputNodesAmount; j++)
                {
                    if (j < _inputNodesAmount)
                        newRecord[0][j] = newTrainingData[i, j];
                    else
                        newRecord[1][j - _inputNodesAmount] = newTrainingData[i, j];
                }
                _data.Add(newRecord);
                _amountOfData++;
            }
        }
    }
}
