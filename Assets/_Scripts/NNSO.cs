using UnityEngine;

[CreateAssetMenu(fileName = "NNSO", menuName = "Neural Network/NNSO")]
public class NNSO : ScriptableObject
{
    public NeuralNetworkData data;

    // Build runtime layers from data
    public NeuralNetworkLayer[] BuildRuntimeLayers()
    {
        if (data == null || data.layers == null || data.layers.Length == 0)
            return null;

        var layers = new NeuralNetworkLayer[data.layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            var ld = data.layers[i];
            layers[i] = NeuralNetworkLayer.FromArrays(
                ld.inputSize,
                ld.outputSize,
                ld.weights,
                ld.biases,
                ld.activation);
        }
        return layers;
    }

    // JSON export/import (we’ll flesh this out later)
    public string ToJson(bool prettyPrint = false)
    {
        return JsonUtility.ToJson(data, prettyPrint);
    }

    public void FromJson(string json)
    {
        data = JsonUtility.FromJson<NeuralNetworkData>(json);
    }
}


[System.Serializable]
public class NeuralNetworkLayerData
{
    public int inputSize;
    public int outputSize;
    public NeuralNetworkLayer.Activation activation;
    public float[] weights; // row-major: [row0..., row1..., ...]
    public float[] biases;
}

[System.Serializable]
public class NeuralNetworkData
{
    public NeuralNetworkLayerData[] layers;
}

public static class NeuralNetworkFactory
{
    public static NeuralNetworkLayer[] BuildLayersFromData(NeuralNetworkData data)
    {
        if (data == null || data.layers == null || data.layers.Length == 0)
            return null;

        var layers = new NeuralNetworkLayer[data.layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            var ld = data.layers[i];
            layers[i] = NeuralNetworkLayer.FromArrays(
                ld.inputSize,
                ld.outputSize,
                ld.weights,
                ld.biases,
                ld.activation);
        }
        return layers;
    }
}