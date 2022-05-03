using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
namespace NeuralNetwork
{
    public class TrainingData
    {
        private readonly int _inputNodesAmount, _outputNodesAmount;
        private readonly int _amountOfRecordsInDataset;
        private readonly List<Vector[]> _data;
        private int _amountOfData;
        private List<List<Vector[]>> _datasets;
        private int _amountOFDatasets;
        public int AmountOfRecordsInDataset { get => _amountOfRecordsInDataset; }
        public List<List<Vector[]>> DataSets { get => _datasets; }
        public int CountDatasets { get => _amountOFDatasets; }
        public TrainingData(Matrix trainingData, int groupBy, int inputNodesAmount, int outputNodesAmount)
        {
            _amountOfRecordsInDataset = groupBy;
            _inputNodesAmount = inputNodesAmount;
            _outputNodesAmount = outputNodesAmount;
            _amountOfData = 0;
            _amountOFDatasets = 0;
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
            DivideDataInDatasets();
        }
        private void DivideDataInDatasets()
        {
            _datasets = new List<List<Vector[]>>();
            for (int i = 0; i < _amountOfData / _amountOfRecordsInDataset; i++)
            {
                List<Vector[]> newDataset = new List<Vector[]>();
                for (int j = 0; j < _amountOfRecordsInDataset; j++)
                {
                    newDataset.Add(_data[i * _amountOfRecordsInDataset + j]);
                }
                _datasets.Add(newDataset);
                _amountOFDatasets++;
            }
        }
    }
}
