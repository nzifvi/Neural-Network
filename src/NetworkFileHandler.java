import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class NetworkFileHandler {
    ArrayList<Request> fileQueue = new ArrayList<>();

    public static class Request{
        final private String filePath;
        private double[] biases = null;
        private double[][][][] filters = null;
        private int learningRate = -1;

        public Request(final String filePath, final double[] biases){
            this.filePath = filePath;
            this.biases = biases;
        }

        public Request(final String filePath, final double[][][][] filters){
            this.filePath = filePath;
            this.filters = filters;
        }

        public Request(final String filePath, final int learningRate){
            this.filePath = filePath;
            this.learningRate = learningRate;
        }

        public double getElementFromFilter(final int no, final int depth, final int row, final int col){
            return filters[no][depth][row][col];
        }

        public double[] getColFromFilter(final int no, final int depth, final int row){
            return filters[no][depth][row];
        }

        public double[][] getFilterSlice(final int no, final int depth){
            return filters[no][depth];
        }

        public double[][][] getFilter(final int no){
            return filters[no];
        }

        public double[][][][] getFilters(){
            return filters;
        }

        public double[] getBiases(){
            return biases;
        }

        public double getBiases(final int index){
            return biases[index];
        }

        public String getFilePath(){
            return filePath;
        }

        public int getLearningRate(){
            return learningRate;
        }

    }

    public void enqueue(final Request request){
        boolean isFound = false;
        int i = 0;
        while(i < fileQueue.size() && !isFound){
            if(fileQueue.get(i).getFilePath().equals(request.filePath)){
                isFound = true;
                fileQueue.set(i, request);
            }
            i++;
        }
        if(!isFound){
            fileQueue.add(request);
        }
    }

    public void save(){
        if(!fileQueue.isEmpty()){
            try{
                for(int i = 0; i < fileQueue.size(); i++){
                    Request request = fileQueue.get(i);
                    FileWriter fw = new FileWriter(loadFile(request.filePath));
                    //Write to file
                    if(request.getFilters() != null){
                        writeFile(fw, request.getFilters());
                    }else if(request.getBiases() != null){
                        writeFile(fw, request.getBiases());
                    }else if(request.getLearningRate() > -1){
                        writeFile(fw, request.getLearningRate());
                    }
                    fw.close();
                }
                fileQueue.clear();
            }catch(Exception e){
                System.out.println("  ! FATAL ERROR: An unknown error has occurred attempting to write to a file");
                e.printStackTrace();
                System.exit(1);
            }
        }else{
            System.out.println("        ! File queue is empty, no saving is required as neural network is up to date");
        }
    }

    public static File loadFile(final String filePath){
        try{
            File file = new File(filePath);
            if(!file.exists()){
                file.createNewFile();
            }
            return file;
        }catch(Exception e){
            System.out.println("  ! FATAL ERROR: An unexpected error occurred when attempting to create " + filePath);
            e.printStackTrace();
            System.exit(1);
        }
        return null;
    }

    public static File loadCheckableFile(final String filePath){
        File file = new File(filePath);
        if(!file.exists()){
            return file;
        }else{
            return null;
        }
    }

    public static void writeFile(FileWriter fw, final double[][][][] filters){
        try{
            for(int no = 0; no < filters.length; no++){
                for(int depth = 0; depth < filters[no].length; depth++){
                    for(int row = 0; row < filters[no][depth].length; row++){
                        for(int col = 0; col < filters[no][depth][row].length; col++){
                            if(col == filters[no][depth][row].length - 1){
                                fw.write(filters[no][depth][row][col] + "");
                            }else{
                                fw.write(filters[no][depth][row][col] + " ");
                            }
                        }
                        fw.write("\n");
                    }
                    fw.write("NEW_DEPTH\n");
                }
                fw.write("NEW_FILTER\n");
            }
            fw.close();
        }catch(Exception e){
            System.out.println("  ! FATAL ERROR: An unknown error has occurred attempting to write to a file");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void writeFile(FileWriter fw, final double[] biases){
        try{
            for(int i = 0; i < biases.length; i++){
                fw.write(biases[i] + "\n");
            }
            fw.close();
        }catch(Exception e){
            System.out.println("  ! FATAL ERROR: An unexpected error occurred when attempting to write to a file");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void writeFile(FileWriter fw, final int learningRate){
        try{
            fw.write(learningRate);
        }catch(Exception e){
            System.out.println("  ! FATAL ERROR: An unexpected error occurred when attempting to write to a file");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static int loadLayerLearningRate(final String filePath){
        File learningRateFile = loadCheckableFile(filePath);
        if(learningRateFile == null){
            System.out.println("    |- -> " + filePath + " cannot be located and is presumed lost, initialising to a default value of 0");
            return 0;
        }else{
            try{
                Scanner scanObj = new Scanner(learningRateFile);
                int learningRate = scanObj.nextInt();
                scanObj.close();
                return learningRate;
            } catch (Exception e) {
                System.out.println("  ! FATAL ERROR: A fatal error has occurred during attempting to read " + filePath + " during the layer's learning rate initialisation");
                e.printStackTrace();
                System.exit(1);
            }
        }
        return -1;
    }

    public static double[][][] loadInput(final String filePath, final int depth, final int row, final int col){
        double[][][] inputArray = new double[depth][row][col];
        File inputFile = loadCheckableFile(filePath);
        if(inputFile == null){
            System.out.println("  ! NON-FATAL ERROR: An input was not able to be read for an undisclosed reason, returning empty matrix");
            return null;
        }else{
            try{
                int nCount = 0;
                int rowCount = 0;
                BufferedReader bufferedReader = new BufferedReader(new FileReader(inputFile));
                String line;
                while ((line = bufferedReader.readLine()) != null && nCount < depth){
                    if(line.equals("n")){
                        nCount++;
                        rowCount = 0;
                    }else{
                        String[] values = line.trim().split("\\s+");
                        double[] matrixRow = Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();
                        inputArray[nCount][row] = matrixRow;
                        rowCount++;
                    }
                }
                bufferedReader.close();
                return inputArray;
            }catch(Exception e){
                System.out.println("  ! FATAL ERROR: A fatal error has occurred during attempting to read " + filePath + " during the loading of an input");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return inputArray;
    }

    public static double[][][][] loadInput(final File file, final int filterNo,  final int filterDepth, final int standardHeight, final int standardWidth) {
        double[][][][] inputArray = new double[filterNo][filterDepth][standardHeight][standardWidth];
        try{
            if (file == null) { //Redo to initialise a 4d matrix of 1s.
                System.out.println("  ! FATAL ERROR: Input file is null. This may potentially be due to a failure to read");
                System.exit(1);
            } else {
                int nCount = 0;
                int aCount = 0;
                int row = 0;
                BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    if (line.equals("n")) {
                        nCount++;
                        row = 0;
                    } else if (line.equals("a")) {
                        aCount++;
                        nCount = 0;
                        row = 0;
                    } else {
                        String[] values = line.split("\\s+");
                        double[] matrixRow = Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();
                        inputArray[aCount][nCount][row] = matrixRow;
                        row++;
                    }
                }
            }
            return inputArray;
        }catch(Exception e){
            System.out.println("  ! FATAL ERROR: An unexpected error occurred during attempting to read " + file + " during the loading of an input");
            e.printStackTrace();
            System.exit(1);
        }
        return null;
    }

    public static double[] loadInput(final File file, final int no){
        double[] inputArray = new double[no];
        try{
            Scanner scanObj = new Scanner(file);
            int i = 0;
            while(i < no && scanObj.hasNext()){
                inputArray[i] = scanObj.nextDouble();
                i++;
            }
            scanObj.close();
            return inputArray;
        }catch(Exception e){
            System.out.println("  ! FATAL ERROR: An unexpected error occurred during attempting to read " + file + " during the loading of an input");
            e.printStackTrace();
            System.exit(1);
        }
        return null;
    }

    public static int[] loadControls(final File file, final int no){
        int[] inputArray = new int[no];
        try{
            Scanner scanObj = new Scanner(file);
            int i = 0;
            while(i < no && scanObj.hasNext()){
                inputArray[i] = scanObj.nextInt();
                i++;
            }
            scanObj.close();
            return inputArray;
        }catch(Exception e){
            System.out.println("  ! FATAL ERROR: An unexpected error occurred during attempting to read " + file + " during the loading of an input");
            e.printStackTrace();
            System.exit(1);
        }
        return null;
    }
}
