import java.io.File;

public class Layer {
    final private int layerDependenciesNum = 5;
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

    public int getLayerDependenciesNum(){
        return layerDependenciesNum;
    }

    public static double[][] createSubArray(double[][] input, final int filterLength, final int row, final int col){
        double[][] subArray = new double[filterLength][filterLength];
        for(int x = 0; x < filterLength; x++){
            for(int y = 0; y < filterLength; y++){
                subArray[x][y] = input[x + row][y + col];
            }
        }
        return subArray;
    }
}

class Convolutor extends Layer {
    private double[][][][] filters;
    private double[][][] inputActvMap;
    private double[][][] outputActvMap;

    private double[] biases;

    public Convolutor(int i){
        super(i);

        initDependencies();
        this.outputActvMap = new double[this.filters.length][][];
    }

    private void initDependencies() {
        int[] controlValues = new int[getLayerDependenciesNum()];
        File file = NetworkFileHandler.loadCheckableFile("dependencies/layers/LayerControls_" + getLayerNum());

        if(file == null){
            System.out.println("  ! FATAL ERROR: The control values for layer " + getLayerNum() + " cannot be found");
            System.exit(1);
        }else{
            controlValues = NetworkFileHandler.loadControls(file, getLayerDependenciesNum());
        }

        final int filterNo = controlValues[0];
        final int filterDepth = controlValues[1];
        final int filterHeight = controlValues[2];
        final int filterWidth = controlValues[3];
        final int biasesNo = controlValues[4];

        initFilters(filterNo, filterDepth, filterHeight, filterWidth);
        initBiases(biasesNo);
    }

    private void initFilters(final int no, final int depth, final int row, final int col){
        double[][][][] loadedFilter = new double[no][depth][row][col];
        File file = NetworkFileHandler.loadCheckableFile("dependencies/layers/layerFilters/filters_Layer" + getLayerNum());
        if(file == null){
            System.out.println("    |- -> Initialising new filter, with default element value of 1, for layer " + getLayerNum());
            for(int currentNo = 0; currentNo < no; currentNo++){
                for(int currentDepth = 0; currentDepth < depth; currentDepth++){
                    for(int currentRow = 0; currentRow < row; currentRow++){
                        for(int currentCol = 0; currentCol < col; currentCol++){
                            loadedFilter[currentNo][currentDepth][currentRow][currentCol] = 1;
                        }
                    }
                }
            }
            this.filters = loadedFilter;
        }else{
            this.filters = NetworkFileHandler.loadInput(file, no, depth, row, col);
        }
    }

    private void initBiases(final int no){
        double[] loadedBiases = new double[no];
        File file = NetworkFileHandler.loadCheckableFile("dependencies/layers/biases_Layer" + getLayerNum());
        if(file == null){
            System.out.println("    |- -> Initialising new biases, with default value of 0, for layer " + getLayerNum());
            for(int currentNo = 0; currentNo < no; currentNo++){
                loadedBiases[currentNo] = 0;
            }
            this.biases = loadedBiases;
        }else{
            this.biases = NetworkFileHandler.loadInput(file, no);
        }
    }

    public static double[][] convolute(double[][] filter, double[][] input, final int learningRate){
        double[][] output = new double[input.length - filter.length + 1][input[0].length - filter[0].length + 1];
        for(int row = 0; row < input.length - filter.length + 1; row++){
            for(int col = 0; col < input[0].length - filter[0].length + 1; col++){
                output[row][col] = getWeightedSums(createSubArray(input, filter.length, row, col), filter);
            }
        }
        return applyActivationFunction(output);
    }

    private static double getWeightedSums(double[][] subArray, double[][] filter){
        double weightedSum = 0;
        for(int row = 0; row < subArray.length; row++){
            for(int col = 0; col < subArray[row].length; col++){
                weightedSum += filter[row][col] * subArray[row][col];
            }
        }
        return weightedSum;
    }

    private static double[][] applyActivationFunction(double[][] array){
        for(int row = 0; row < array.length; row++){
            for(int col = 0; col < array[row].length; col++){
                array[row][col] = NetworkMathHandler.TANH_Activation(array[row][col]);
            }
        }
        return array;
    }
}

class Pool extends Layer {

}
