using UnityEngine;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

public class NeuralNetwork : MonoBehaviour
{
    [SerializeField] private bool UsePregeneratedNetwork = false;

    [SerializeField] private NNSO _source;
    public NNSO Source { get { return _source; } set{ _source = value; } }

    [SerializeField] private int InputSize;
    [SerializeField] private int OutputSize;
    [SerializeField] private int HiddenLayerCount;
    [SerializeField] private int HiddenLayerSize;


    private NeuralNetworkLayer[] Layers;

    #region Initialization
    
    public void SetupDimensions(int inputSize,  int outputSize, int hiddenLayerCount, int hiddenLayerSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        HiddenLayerCount = hiddenLayerCount;
        HiddenLayerSize = hiddenLayerSize;
    }
    
    public void InitializeNN(NNSO network)
    {
        UsePregeneratedNetwork = true;
        _source = network;
        InitializeNN();
    }

    public void InitializeNN()
    {
        if (UsePregeneratedNetwork)
        {
            Layers = _source.BuildRuntimeLayers();
            return;
        }

        InitializeRandomNetwork();

    }

    private void InitializeRandomNetwork()
    {
        if (HiddenLayerCount <= 0)
        {
            Layers = new NeuralNetworkLayer[1];
            Layers[0] = CreateRandomLayer(InputSize, OutputSize, NeuralNetworkLayer.Activation.Sigmoid);
        }
        else if (HiddenLayerCount == 1)
        {
            Layers = new NeuralNetworkLayer[2];
            Layers[0] = CreateRandomLayer(InputSize, HiddenLayerSize, NeuralNetworkLayer.Activation.ReLU);
            Layers[1] = CreateRandomLayer(HiddenLayerSize, OutputSize, NeuralNetworkLayer.Activation.Sigmoid);
        }
        else
        {
            Layers = new NeuralNetworkLayer[HiddenLayerCount + 1];
            Layers[0] = CreateRandomLayer(InputSize, HiddenLayerSize, NeuralNetworkLayer.Activation.ReLU);

            for (int l = 1; l < HiddenLayerCount; l++)
            {
                Layers[l] = CreateRandomLayer(HiddenLayerSize, HiddenLayerSize, NeuralNetworkLayer.Activation.ReLU);
            }

            Layers[HiddenLayerCount] = CreateRandomLayer(HiddenLayerSize, OutputSize, NeuralNetworkLayer.Activation.Sigmoid);
        }
    }

    private NeuralNetworkLayer CreateRandomLayer(int inputSize, int outputSize, NeuralNetworkLayer.Activation activation)
    {
        var biases = new float[outputSize];
        var weights = new float[outputSize * inputSize];

        for (int o = 0; o < outputSize; o++)
        {
            biases[o] = Random.Range(-0.5f, 0.5f);

            for (int i = 0; i < inputSize; i++)
            {
                int idx = o * inputSize + i;
                weights[idx] = Random.Range(-1f, 1f);
            }
        }

        return NeuralNetworkLayer.FromArrays(inputSize, outputSize, weights, biases, activation);
    }
    #endregion

    public NeuralNetworkData ToData()
    {
        var data = new NeuralNetworkData();
        data.layers = new NeuralNetworkLayerData[Layers.Length];

        for (int i = 0; i < Layers.Length; i++)
        {
            var layer = Layers[i];
            data.layers[i] = new NeuralNetworkLayerData
            {
                inputSize = layer.InputSize,  
                outputSize = layer.OutputSize,
                activation = layer.Active,      
                weights = layer.GetWeightsFlat(),
                biases = layer.GetBiasesArray()
            };
        }

        return data;
    }

    private static readonly VectorBuilder<float> V = Vector<float>.Build;

    public Vector<float> Forward(float[] inputArray)
    {
        if (inputArray.Length != InputSize)
            throw new System.ArgumentException($"Expected {InputSize} inputs, got {inputArray.Length}");

        var input = V.DenseOfArray(inputArray);
        return Forward(input);
    }

    public Vector<float> Forward(Vector<float> input)
    {
        if (Layers == null || Layers.Length == 0)
            throw new System.InvalidOperationException("Neural network not initialized.");

        var x = input;
        for (int i = 0; i < Layers.Length; i++)
            x = Layers[i].CalculateLayer(x);
        return x;
    }

}
