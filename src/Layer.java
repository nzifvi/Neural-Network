import java.io.File;
import java.io.FileWriter;

public class Layer {
    private int layerDependenciesNum;
    private int layerNum;
    private int learningRate;

    public Layer(final int i){
        layerNum = i;
        initLearningRate();
    }

    private void initLearningRate(){
        File learningRateFile = NetworkFileHandler.loadFile("dependencies/layers/learningRate_Layer" + layerNum);
        if(learningRateFile == null){
            System.out.println("    |- -> Layer learning rate file lost, initialising to zero as a default value");
            learningRate = 0;
        }else{
            learningRate = NetworkFileHandler.loadLayerLearningRate("dependencies/layers/learningRate_Layer" + layerNum);
        }
    }

    public int getLearningRate(){
        return learningRate;
    }

    public void setLearningRate(final int learningRate){
        this.learningRate = learningRate;
    }

    public int getLayerNum(){
        return layerNum;
    }

    public void setLayerNum(final int layerNum){
        this.layerNum = layerNum;
    }
}

class Convolutor extends Layer {
    private double[][][][] filter;
    private double[][][] inputActvMap;
    private double[][][] outputActvMap;

    private double[] biases;

    public Convolutor(int i){
        super(i);


    }
}

class Pool extends Layer {

}
