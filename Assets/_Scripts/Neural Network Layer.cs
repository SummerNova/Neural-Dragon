using UnityEngine;
using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;


public class NeuralNetworkLayer
{
    private readonly int _inputSize;
    public int InputSize { get { return _inputSize; } }
    private readonly int _outputSize;
    public int OutputSize {  get { return _outputSize; } }

    private readonly Matrix<float> Weights;
    private readonly Vector<float> Biases;
    private readonly Activation _active;
    public Activation Active {  get { return _active; } }

    public NeuralNetworkLayer(int inputSize, int outputSize, List<float> weights, List<float> biases, Activation active)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        Weights = Matrix<float>.Build.Dense(outputSize, inputSize);
        Biases = Vector<float>.Build.Dense(outputSize);

        for (int k = 0; k < outputSize; k++)
        {
            if (k < biases.Count)
                Biases[k] = biases[k];
            for (int n = 0; n < inputSize; n++)
            {
                if (k * inputSize + n < weights.Count)
                    Weights[k, n] = weights[k * inputSize + n];
            }
        }
        _active = active;
    }

    public static NeuralNetworkLayer FromArrays(
    int inputSize,
    int outputSize,
    float[] weights,
    float[] biases,
    Activation activation)
    {
        if (weights.Length != outputSize * inputSize)
            throw new ArgumentException("Wrong weights length", nameof(weights));
        if (biases.Length != outputSize)
            throw new ArgumentException("Wrong biases length", nameof(biases));

        return new NeuralNetworkLayer(
            inputSize,
            outputSize,
            new List<float>(weights),
            new List<float>(biases),
            activation);
    }

    public Vector<float> CalculateLayer(Vector<float> Inputs)
    {
        if (Inputs.Count != _inputSize) throw new IndexOutOfRangeException("Inputs Length does not match Input Size for NN Layer");

        Vector<float> Output = Weights * Inputs + Biases;

        Output = _active switch
        {
            Activation.ReLU => Output.Map(ReLU),
            Activation.Sigmoid => Output.Map(Sigmoid),
            Activation.Tanh => Output.Map(Tanh),
            Activation.Softmax => Softmax(Output),
            _ => throw new InvalidOperationException($"Unsupported activation: {_active}"),
        };

        return Output;
    }
    public float[] GetWeightsFlat()
    {
        var arr = new float[_outputSize * _inputSize];
        int i = 0;
        for (int k = 0; k < _outputSize; k++)
            for (int n = 0; n < _inputSize; n++)
                arr[i++] = Weights[k, n];
        return arr;
    }

    public float[] GetBiasesArray()
    {
        var arr = new float[_outputSize];
        for (int k = 0; k < _outputSize; k++)
            arr[k] = Biases[k];
        return arr;
    }

    #region Activations
    private float Sigmoid(float x) => 1f/(1f + Mathf.Exp(-x));
    private float Tanh(float x)
    {
        float e = Mathf.Exp(2f * x);
        return (e - 1f) / (e + 1f);
    }
    private float ReLU(float x) => Mathf.Max(0f, x);
    public static Vector<float> Softmax(Vector<float> v)
    {
        var max = v.Maximum();          
        var exp = v.Map(x => Mathf.Exp(x - max));
        var sum = exp.Sum();
        return exp / sum;
    }

    public enum Activation { Sigmoid, Tanh, ReLU, Softmax}
    #endregion
}