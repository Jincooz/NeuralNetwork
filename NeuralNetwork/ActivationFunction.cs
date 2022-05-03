using Numerics = MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
namespace NeuralNetwork
{
    public interface ActivationFunction
    {
        Vector ActivationFunction(Vector value);
        Vector ActivationFunctionDerivative(Vector value);
    }
    public readonly struct ActivationFunctionStrategy
    {
        public static ActivationFunctionIndentity Indentity { get; }
        public static ActivationFunctionBinaryStep BinaryStep { get; }
        public static ActivationFunctionLogistic Logistic { get; }
        public static ActivationFunctionTanh Tanh { get; }
    }
    public class ActivationFunctionIndentity : ActivationFunction
    {
        public Vector ActivationFunction(Vector value) => value;
        public Vector ActivationFunctionDerivative(Vector value) => Vector.Build.Dense(value.Count, 1.0);
    }
    public class ActivationFunctionBinaryStep : ActivationFunction
    {
        public Vector ActivationFunction(Vector value)
        {
            for (int i = 0; i < value.Count; i++)
            {
                value[i] = (value[i] >= 0 ? 1 : 0);
            }
            return value;
        }
        public Vector ActivationFunctionDerivative(Vector value)
        {
            for (int i = 0; i < value.Count; i++)
            {
                if (value[i] == 0) throw new InvalidOperationException("No derivative in 0");
                value[i] = 0;
            }
            return value;
        }
    }
    public class ActivationFunctionLogistic : ActivationFunction
    {
        public Vector ActivationFunction(Vector value)
        {
            for (int i = 0; i < value.Count; i++)
            {
                value[i] = 1 / (1 + Math.Exp(-value[i]));
            }
            return value;
        }
        public Vector ActivationFunctionDerivative(Vector value)
        {
            for (int i = 0; i < value.Count; i++)
            {
                value[i] = (1 / (1 + Math.Exp(-value[i]))) * (1 - 1 / (1 + Math.Exp(-value[i])));
            }
            return value;
        }
    }
    public class ActivationFunctionTanh : ActivationFunction
    {
        public Vector ActivationFunction(Vector value)
        {
            for (int i = 0; i < value.Count; i++)
            {
                value[i] = Math.Tanh(value[i]);
            }
            return value;
        }
        public Vector ActivationFunctionDerivative(Vector value)
        {
            for (int i = 0; i < value.Count; i++)
            {
                value[i] = 1 - Math.Pow(Math.Tanh(value[i]), 2);
            }
            return value;
        }
    }
}
